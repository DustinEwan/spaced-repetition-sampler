import math
import random
import heapq
from torch.utils.data import Sampler


class ExponentialMovingStats:
    """
    Tracks the exponential moving average of 'x' and 'x^2',
    allowing us to compute a running standard deviation.
    """
    def __init__(self, alpha=0.1):
        """
        Args:
            alpha: smoothing factor for EMA. 
        """
        self.alpha = alpha
        self.mean = 0.0
        self.mean_sq = 0.0
        self.initialized = False

    def update(self, x: float):
        if not self.initialized:
            self.mean = x
            self.mean_sq = x * x
            self.initialized = True
        else:
            self.mean = self.alpha * x + (1 - self.alpha) * self.mean
            self.mean_sq = self.alpha * (x ** 2) + (1 - self.alpha) * self.mean_sq

    @property
    def std(self):
        var = self.mean_sq - self.mean**2
        return math.sqrt(var) if var > 0 else 0.0

class SpacedRepetitionSampler(Sampler):
    """
    A custom sampler that uses a single priority queue (heap) for seen samples
    and a set for unseen samples. In addition to scheduling samples based on
    their measured difficulty (loss) and a base spacing, we also track a per–sample
    interval. If a sample is answered "easily" (i.e. its loss is below a success threshold),
    its interval is increased so that its cooldown grows longer over repeated successes.
    If a sample is "missed" (loss above threshold), its interval resets.
    
    Sampling for each batch reserves a fraction from the unseen (backlog) samples and
    fills the remainder by popping the best candidates from the heap.
    """

    def __init__(
        self,
        data_source,
        loss_stats,  # an instance of ExponentialMovingStats
        num_steps,
        batch_size,
        spacing_steps=0,
        seed=42,
        growth_rate=1.2,        # multiplicative factor for "successful" samples
        max_interval=100,       # maximum allowed interval multiplier
        success_threshold_std=0.5  # success if loss is less than mean + success_threshold_std * std
    ):
        """
        Args:
            data_source: Typically an IndexedDataset that returns (x, y, idx).
            loss_stats: An ExponentialMovingStats object tracking mean and std of losses.
            num_steps: Total number of batches (iterations) to yield.
            batch_size: Number of samples per batch.
            spacing_steps: Base spacing in steps; if 0, defaults to num_steps // 1000.
            seed: Random seed.
            growth_rate: How much to increase a sample's interval on a successful (easy) review.
            max_interval: Maximum allowed interval multiplier.
            success_threshold_std: A sample is considered "successful" if its updated loss is less than
                                   mean + (success_threshold_std * std).
        """
        super().__init__(data_source)
        self.data_source = data_source
        self.num_samples = len(data_source)
        self.loss_stats = loss_stats
        
        # If spacing_steps is not provided, default to roughly 0.1% of num_steps (at least 1).
        self.spacing_steps = spacing_steps if spacing_steps > 0 else max(1, num_steps // 1000)
        self.batch_size = batch_size
        self.num_steps = num_steps

        self.growth_rate = growth_rate
        self.max_interval = max_interval
        self.success_threshold_std = success_threshold_std

        random.seed(seed)

        # For each sample, store its most recent difficulty; None means “unseen.”
        self.difficulties = [None] * self.num_samples

        # For bookkeeping: last step this sample was drawn.
        self.last_drawn_step = [-math.inf] * self.num_samples

        # Unseen (backlog) samples – a set of sample indices.
        self.unseen_samples = set(range(self.num_samples))

        # Heap (priority queue) for seen samples.
        # Each entry is a tuple: (next_available, -difficulty, sample_idx)
        # We also keep a dictionary for current valid entries.
        self.heap = []
        self.heap_entries = {}

        # Per–sample interval (multiplier). Initially, every sample is at interval 1.0.
        self.intervals = [1.0] * self.num_samples

        self.current_step = 0
        self.last_batch_indices = None

    def __len__(self):
        return self.num_steps

    def __iter__(self):
        """
        For each training step, construct a batch as follows:
          1. Reserve a variable fraction (using a sine–curve, with at least backlog_ratio)
             from the unseen set.
          2. Fill the remainder from the heap of seen samples.
             (If no sample is eligible, force-pop the top element.)
          3. If needed and unseen samples remain, fill additional slots from unseen.
        """
        for step in range(self.num_steps):
            self.current_step = step
            batch_indices = []

            # Calculate a varying ratio (between 0 and 1) for backlog usage.
            ratio = math.sin(step / (self.spacing_steps * 2)) / 2 + 0.5
            backlog_needed = int(ratio * self.batch_size)

            # 1. Pull from the unseen backlog.
            unseen_list = list(self.unseen_samples)
            num_backlog = min(backlog_needed, len(unseen_list))
            chosen_backlog = unseen_list[:num_backlog]
            for idx in chosen_backlog:
                self.unseen_samples.remove(idx)
            batch_indices.extend(chosen_backlog)

            needed = self.batch_size - len(batch_indices)

            # 2. Fill from the heap of seen samples.
            while needed > 0:
                if self.heap:
                    next_avail, neg_diff, sample_idx = self.heap[0]
                    if next_avail <= self.current_step:
                        # Sample is eligible.
                        heapq.heappop(self.heap)
                        if sample_idx in self.heap_entries:
                            del self.heap_entries[sample_idx]
                        batch_indices.append(sample_idx)
                        needed -= 1
                    else:
                        # No sample is yet eligible; force-pop the top.
                        next_avail, neg_diff, sample_idx = heapq.heappop(self.heap)
                        if sample_idx in self.heap_entries:
                            del self.heap_entries[sample_idx]
                        batch_indices.append(sample_idx)
                        needed -= 1
                else:
                    # 3. If heap is empty but unseen samples remain, fill from unseen.
                    if self.unseen_samples:
                        unseen_list = list(self.unseen_samples)
                        take = min(needed, len(unseen_list))
                        for idx in unseen_list[:take]:
                            self.unseen_samples.remove(idx)
                        batch_indices.extend(unseen_list[:take])
                        needed -= take
                    else:
                        break

            assert len(batch_indices) == self.batch_size, (
                f"Batch incomplete: got {len(batch_indices)} instead of {self.batch_size}"
            )

            # Record that these samples were drawn at the current step.
            for idx in batch_indices:
                self.last_drawn_step[idx] = self.current_step

            self.last_batch_indices = batch_indices
            yield batch_indices

    def update_difficulties(self, batch_loss):
        """
        After processing a batch, update each sample's difficulty and personal interval,
        then compute its new next–available time and push it back into the heap.
        
        The process is as follows:
          - Update the global loss stats.
          - For each sample in the batch:
              * Add noise to its reported loss.
              * Compare its updated loss (difficulty) with the global mean and std.
              * If the sample's loss is below (mean + success_threshold_std * std), consider it "easy"
                and increase its personal interval (up to max_interval).
              * Otherwise, reset its interval to 1.0.
              * Compute the new next–available time as:
                    current_step + spacing_steps * personal_interval * base_multiplier,
                where base_multiplier is computed from the sample’s difficulty relative to global stats.
        """
        # Update global loss stats.
        self.loss_stats.update(float(batch_loss))
        mean_val = self.loss_stats.mean
        std_val = self.loss_stats.std

        for idx in self.last_batch_indices:
            # Add noise so that samples do not always receive the same updated difficulty.
            noise = random.gauss(0, std_val / 2) if std_val > 0 else 0
            new_diff = float(batch_loss) + noise
            self.difficulties[idx] = new_diff

            # Remove from unseen (if present) since it is now seen.
            self.unseen_samples.discard(idx)

            # Update the per–sample interval based on review outcome.
            # Here, if the sample's loss is lower than mean + success_threshold_std * std,
            # we consider it "successful" and increase its interval.
            if std_val > 0 and new_diff < (mean_val + self.success_threshold_std * std_val):
                self.intervals[idx] = min(self.intervals[idx] * self.growth_rate, self.max_interval)
            else:
                self.intervals[idx] = 1.0

            # Compute a base multiplier from the current difficulty.
            base_multiplier = self._spacing_multiplier(new_diff, mean_val, std_val)
            
            # Add some additional noise of ±25% of spacing_steps.
            noise_steps = random.randint(-self.spacing_steps // 4, self.spacing_steps // 4)
            new_next_available = self.current_step + int(self.spacing_steps * self.intervals[idx] * base_multiplier) + noise_steps

            # Create the new heap entry and push it.
            entry = (new_next_available, -new_diff, idx)
            heapq.heappush(self.heap, entry)
            self.heap_entries[idx] = entry

    def _spacing_multiplier(self, diff, mean_val, std_val):
        """
        Returns a base multiplier for spacing based on the sample's difficulty relative to global loss stats.
        For example:
          - If diff < mean - 2*std: multiplier = 4.0 (easiest sample → long wait)
          - If diff < mean - 1*std: multiplier = 2.0
          - If diff < mean + 1*std: multiplier = 1.0
          - If diff < mean + 2*std: multiplier = 0.5
          - Otherwise: multiplier = 0.25 (hardest sample → short wait)
        """
        if std_val == 0:
            return 1.0
        if diff < mean_val - 2 * std_val:
            return 4.0
        elif diff < mean_val - 1 * std_val:
            return 2.0
        elif diff < mean_val + 1 * std_val:
            return 1.0
        elif diff < mean_val + 2 * std_val:
            return 0.5
        else:
            return 0.25


