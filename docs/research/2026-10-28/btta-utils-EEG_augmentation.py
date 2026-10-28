import torch
import numpy as np
import random
from numpy.random import RandomState
from mne.filter import notch_filter


# 1. Identity Transform (no change)
class IdentityTransform:
    def apply(self, eeg_data):
        return eeg_data


# 2. Gaussian Noise Augmentation
class GaussianNoise:
    def __init__(self, std, num_aug=10):
        self.std = std
        self.num_aug = num_aug

    def apply(self, eeg_data):
        augmented_data = []
        for _ in range(self.num_aug):
            noise = torch.randn_like(eeg_data) * self.std
            augmented_data.append(eeg_data + noise)
        return augmented_data


# 3. Time Shift Augmentation
class TimeShift:
    def __init__(self, max_delta_time, num_aug=10):
        self.max_delta_time = max_delta_time
        self.num_aug = num_aug

    def apply(self, eeg_data):
        augmented_data = []
        for _ in range(self.num_aug):
            shift = random.randint(-self.max_delta_time, self.max_delta_time)
            augmented_data.append(torch.roll(eeg_data, shifts=shift, dims=-1))
        return augmented_data


# 4. Frequency Shift Augmentation
class FrequencyShift:
    def __init__(self, max_delta_freq, sfreq=250, num_aug=10):
        self.max_delta_freq = max_delta_freq
        self.sfreq = sfreq
        self.num_aug = num_aug

    def apply(self, eeg_data):
        augmented_data = []
        for _ in range(self.num_aug):
            freq_shift = np.random.uniform(-self.max_delta_freq, self.max_delta_freq)
            fft_data = torch.fft.rfft(eeg_data, dim=-1)
            freqs = torch.fft.rfftfreq(eeg_data.shape[-1], d=1.0 / self.sfreq)
            shift = torch.exp(2j * np.pi * freq_shift * freqs).to(eeg_data.device)
            shifted_fft = fft_data * shift
            shifted_data = torch.fft.irfft(shifted_fft, dim=-1)
            augmented_data.append(shifted_data)
        return augmented_data


# 5. FTSurrogate Augmentation 生成伪替代信号，通过添加随机相位噪声。
class FTSurrogate:
    """
    FT surrogate augmentation of a single EEG channel.
        phase_noise_magnitude: float
            Float between 0 and 1 setting the range over which the phase
            perturbation is uniformly sampled:
            [0, `phase_noise_magnitude` * 2 * `pi`].
        channel_indep : bool
            Whether to sample phase perturbations independently for each channel or
            not. It is advised to set it to False when spatial information is
            important for the task, like in BCI.
    """

    def __init__(self, phase_noise_magnitude, channel_indep=False, num_aug=10):
        self.phase_noise_magnitude = phase_noise_magnitude
        self.channel_indep = channel_indep
        self.num_aug = num_aug

    def apply(self, eeg_data):
        fft_data = torch.fft.rfft(eeg_data, dim=-1)
        phase = torch.angle(fft_data)
        mag = torch.abs(fft_data)

        augmented_data = []
        for _ in range(self.num_aug):
            if self.channel_indep:
                # Independent phase noise for each channel
                random_phase_noise = torch.rand_like(phase) * 2 * np.pi * self.phase_noise_magnitude
            else:
                # Shared phase noise across all channels
                random_phase_noise = torch.rand(1, phase.shape[-1],
                                                device=phase.device) * 2 * np.pi * self.phase_noise_magnitude
                random_phase_noise = random_phase_noise.expand_as(phase)

            new_phase = phase + random_phase_noise
            new_fft_data = mag * torch.exp(1j * new_phase)

            augmented_data.append(torch.fft.irfft(new_fft_data, dim=-1))
        return augmented_data


# 6. SmoothTimeMask Augmentation
class SmoothTimeMask:
    """
    Smoothly replace a contiguous part of all channels by zeros.
        mask_start_per_sample : torch.tensor
            Tensor of integers containing the position (in last dimension) where to
            start masking the signal. Should have the same size as the first
            dimension of X (i.e. one start position per example in the batch).
        mask_len_samples : int
            Number of consecutive samples to zero out.
    """

    def __init__(self, mask_len_samples, num_aug=10):
        self.mask_len_samples = mask_len_samples
        self.num_aug = num_aug

    def apply(self, eeg_data):
        batch_size, _, n_channels, seq_len = eeg_data.shape
        device = eeg_data.device

        augmented_data = []
        for _ in range(self.num_aug):
            # Generate random start positions for each sample in the batch
            mask_start_per_sample = torch.randint(0, seq_len - self.mask_len_samples, (batch_size,), device=device)
            mask = self._create_smooth_mask(eeg_data, mask_start_per_sample)
            augmented_data.append(eeg_data * mask)

        return augmented_data

    def _create_smooth_mask(self, eeg_data, mask_start_per_sample):
        batch_size, _, n_channels, seq_len = eeg_data.shape
        device = eeg_data.device

        # Create a time axis for the mask
        t = torch.arange(seq_len, device=device).float().unsqueeze(0).repeat(batch_size, 1, n_channels, 1)

        # Broadcast mask_start_per_sample to align with the data shape
        mask_start_per_sample = mask_start_per_sample.view(-1, 1, 1, 1)

        # Smooth sigmoid transition scale factor
        s = 1000 / seq_len

        # Create the smooth mask using sigmoid transitions
        mask = (
                torch.sigmoid(s * -(t - mask_start_per_sample)) +
                torch.sigmoid(s * (t - mask_start_per_sample - self.mask_len_samples))
        )
        return mask


# 7. Channels Dropout Augmentation
class ChannelsDropout:
    """
    Randomly set channels to flat signal.
    p_drop : float
        Float between 0 and 1 setting the probability of dropping each channel.
    random_state : int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Defaults to None.
    """

    def __init__(self, p_drop, random_state=None, num_aug=10):
        self.p_drop = p_drop
        self.random_state = RandomState(random_state) if random_state is not None else np.random
        self.num_aug = num_aug

    def _pick_channels_randomly(self, eeg_data, p_pick):
        """
        Generate a binary mask indicating which channels to shuffle.

        Parameters
        ----------
        batch_size: int
            The number of samples in the batch.
        n_channels: int
            The number of channels in each sample.
        p_pick: float
            The probability of picking each channel.

        Returns
        -------
        torch.Tensor
            A binary mask of shape (batch_size, n_channels) indicating the channels to shuffle.
        """
        # Generate uniform random values
        batch_size, _, n_channels, _ = eeg_data.shape
        unif_samples = torch.as_tensor(
            self.random_state.uniform(0, 1, size=(batch_size, n_channels)),
            dtype=torch.float,
            device=eeg_data.device
        )
        # Use sigmoid to create binary mask
        return torch.sigmoid(1000 * (unif_samples - p_pick))

    def apply(self, eeg_data):
        augmented_data = []
        for _ in range(self.num_aug):
            mask = self._pick_channels_randomly(eeg_data, self.p_drop)
            augmented_data.append(eeg_data * mask.unsqueeze(1).unsqueeze(-1))
        return augmented_data


# 8. Channels Shuffle Augmentation
class ChannelsShuffle:
    """
    Randomly shuffle channels in EEG data matrix.
        p_shuffle: float | None
            Float between 0 and 1 setting the probability of including the channel
            in the set of permutted channels.

    """

    def __init__(self, p_shuffle, random_state=None, num_aug=10):
        self.p_shuffle = p_shuffle
        self.random_state = RandomState(random_state) if random_state is not None else np.random
        self.num_aug = num_aug

    def _pick_channels_randomly(self, eeg_data, p_pick):
        """
        Generate a binary mask indicating which channels to shuffle.

        Parameters
        ----------
        batch_size: int
            The number of samples in the batch.
        n_channels: int
            The number of channels in each sample.
        p_pick: float
            The probability of picking each channel.

        Returns
        -------
        torch.Tensor
            A binary mask of shape (batch_size, n_channels) indicating the channels to shuffle.
        """
        # Generate uniform random values
        batch_size, _, n_channels, _ = eeg_data.shape
        unif_samples = torch.as_tensor(
            self.random_state.uniform(0, 1, size=(batch_size, n_channels)),
            dtype=torch.float,
            device=eeg_data.device
        )
        # Use sigmoid to create binary mask
        return torch.sigmoid(1000 * (unif_samples - p_pick))

    def _make_permutation_matrix(self, batch_size, n_channels, mask):
        """
        Generate a permutation matrix for channel shuffling.

        mask: torch.Tensor
            Binary mask indicating which channels to shuffle.

        -------
        torch.Tensor
            Permutation matrix of shape (batch_size, n_channels, n_channels).
        """
        batch_permutations = torch.empty(batch_size, 1, n_channels, n_channels, device=mask.device)

        for b, mask_b in enumerate(mask):
            channels_to_shuffle = torch.arange(n_channels, device=mask.device)
            channels_to_shuffle = channels_to_shuffle[mask_b.bool()]

            if len(channels_to_shuffle) > 1:
                # Shuffle the selected channels
                shuffled_indices = torch.randperm(len(channels_to_shuffle))
                channels_permutation = torch.arange(n_channels, device=mask.device)
                channels_permutation[channels_to_shuffle] = channels_to_shuffle[shuffled_indices]
            else:
                channels_permutation = torch.arange(n_channels, device=mask.device)

            # Convert the permutation to a one-hot encoded matrix
            batch_permutations[b, ...] = torch.eye(n_channels, device=mask.device)[channels_permutation]

        return batch_permutations

    def apply(self, eeg_data):
        batch_size, _, n_channels, _ = eeg_data.shape

        augmented_data = []
        for _ in range(self.num_aug):
            # Generate a binary mask for each sample, where 1 indicates the channel will be shuffled
            mask = self._pick_channels_randomly(eeg_data, 1 - self.p_shuffle)

            # Generate the permutation matrix for the batch
            permutation_matrix = self._make_permutation_matrix(batch_size, n_channels, mask)

            # Use matrix multiplication to shuffle the channels
            shuffled_eeg_data = torch.matmul(permutation_matrix, eeg_data)

            augmented_data.append(shuffled_eeg_data)

        return augmented_data


# 9. Bandstop Filter Augmentation
class BandstopFilter:
    """Apply a band-stop filter with desired bandwidth at a randomly selected
    frequency position between 0 and max_freq.
    bandwidth : float
        Bandwidth of the filter, i.e., the distance between the low and high cut frequencies.
    sfreq : float
        Sampling frequency of the signals to be filtered.
    max_freq : float
        Maximal admissible frequency. The low cut frequency will be sampled so
        that the corresponding high cut frequency + transition (=1Hz) are below
        ``max_freq``. If omitted or `None`, will default to the Nyquist
        frequency (``sfreq / 2``).
    """

    def __init__(self, probability, bandwidth, sfreq, max_freq=None, num_aug=10, random_state=None):
        self.probability = probability
        self.bandwidth = bandwidth
        self.sfreq = sfreq
        self.max_freq = max_freq or sfreq / 2
        self.num_aug = num_aug
        self.random_state = RandomState(random_state) if random_state is not None else np.random

        assert self.bandwidth < self.max_freq, "Bandwidth must be smaller than max_freq."

    def _get_notch_freq(self):
        """Randomly select notch frequency within valid range."""
        low = 1 + 2 * self.bandwidth
        high = self.max_freq - 1 - 2 * self.bandwidth
        return self.random_state.uniform(low, high)

    def apply(self, eeg_data):
        augmented_data = []
        for _ in range(self.num_aug):
            transformed_eeg_data = eeg_data.clone()
            for i, sample in enumerate(transformed_eeg_data):
                if np.random.rand() <= self.probability:  # Apply filter based on probability
                    notch_freq = self._get_notch_freq()
                    sample_np = sample.cpu().numpy().astype(np.float64)
                    filtered_sample = notch_filter(
                        sample_np,
                        Fs=self.sfreq,
                        freqs=notch_freq,
                        method="fir",
                        notch_widths=self.bandwidth,
                        verbose=False
                    )
                    transformed_eeg_data[i] = torch.as_tensor(filtered_sample, device=eeg_data.device)
            augmented_data.append(transformed_eeg_data)

        return augmented_data


# 10. Channels Symmetry Augmentation
class ChannelsSymmetry:
    """
        Permute EEG channels inverting left and right-side sensors.

        Parameters
        ----------
        ordered_ch_names : list
            Ordered list of strings containing the names (in 10-20 nomenclature) of the EEG channels
            that will be transformed. Each channel name should correspond to the correct position in the EEG data.
        """

    def __init__(self, ordered_ch_names=None, symmetry_prob=0.5, num_aug=10):
        if ordered_ch_names is None:
            ordered_ch_names = ['EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5', 'EEG-C3', 'EEG-6',
                                'EEG-Cz', 'EEG-7', 'EEG-C4', 'EEG-8', 'EEG-9', 'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13',
                                'EEG-14', 'EEG-Pz', 'EEG-15', 'EEG-16']
        assert isinstance(ordered_ch_names, list) and all(isinstance(ch, str) for ch in ordered_ch_names), \
            "ordered_ch_names should be a list of strings."
        self.ordered_ch_names = ordered_ch_names
        self.symmetry_prob = symmetry_prob
        self.pairs = [
            (1, 5), (2, 4), (6, 12), (7, 11), (8, 10), (13, 17), (14, 16), (18, 20)
        ]  # Pairs to swap relative to the symmetry axis
        self.num_aug = num_aug

    def _create_symmetry_permutation(self):
        """
        Create the permutation list to map each channel to its symmetric counterpart based on predefined axes.

        Returns
        -------
        permutation : list of int
            List containing the indices of the channels after symmetry inversion.
        """
        permutation = list(range(len(self.ordered_ch_names)))  # Start with identity mapping

        for left, right in self.pairs:
            if random.random() < self.symmetry_prob:
                # Swap the left and right pairs
                permutation[left], permutation[right] = permutation[right], permutation[left]

        return permutation

    def apply(self, eeg_data):
        """
        Apply the channel symmetry flip to the EEG data.

        Parameters
        ----------
        eeg_data : torch.Tensor
            Input EEG data of shape (batch_size, n_channels, n_samples).

        Returns
        -------
        torch.Tensor
            EEG data with channels permuted according to symmetry.
        """
        augmented_data = []
        for _ in range(self.num_aug):
            permutation = torch.tensor(self._create_symmetry_permutation(), device=eeg_data.device)
        # Permute the channels based on the computed symmetry mapping
            permuted_eeg_data = eeg_data[:, :, permutation, :]
            augmented_data.append(permuted_eeg_data)
        return augmented_data


if __name__ == '__main__':
    # Test the augmentations
    eeg_data = torch.randn(2, 1, 22, 250)
    augmentations = [
        IdentityTransform(),
        GaussianNoise(std=0.16, num_aug=2),
        TimeShift(max_delta_time=25, num_aug=2),
        FrequencyShift(max_delta_freq=2, sfreq=250, num_aug=2),
        FTSurrogate(phase_noise_magnitude=1, num_aug=2),
        SmoothTimeMask(mask_len_samples=30, num_aug=2),
        ChannelsDropout(p_drop=0.2, num_aug=2),
        ChannelsShuffle(p_shuffle=0.6, num_aug=2),
        BandstopFilter(probability=0.9, bandwidth=1.25, sfreq=250, num_aug=2),
        ChannelsSymmetry(symmetry_prob=0.5, num_aug=2)
    ]
    aug_data = []
    for aug in augmentations:
        augmented_data = aug.apply(eeg_data)
        if isinstance(augmented_data, list):
            augmented_data = torch.cat(augmented_data, dim=0)
        aug_data.append(augmented_data)
        # print(aug.__class__.__name__, augmented_data.shape)
    aug_data = torch.cat(aug_data, dim=0)
    print("Augmentations applied successfully!")