use std::cmp::min;

use polars_arrow::bitmap::MutableBitmap;
use polars_utils::slice::load_padded_le_u64;

/// Bit array with a filter to speed up searching for set bits when sparse,
/// based on section 4.1 from Khayyat et al. 2015,
/// "Lightning Fast and Space Efficient Inequality Joins"
pub struct FilteredBitArray {
    bit_array: MutableBitmap,
    filter: MutableBitmap,
}

impl FilteredBitArray {
    const CHUNK_SIZE: usize = 1024;

    pub fn from_len_zeroed(len: usize) -> Self {
        Self {
            bit_array: MutableBitmap::from_len_zeroed(len),
            filter: MutableBitmap::from_len_zeroed(len.div_ceil(Self::CHUNK_SIZE)),
        }
    }

    pub unsafe fn set_bit_unchecked(&mut self, index: usize) {
        self.bit_array.set_unchecked(index, true);
        self.filter.set_unchecked(index / Self::CHUNK_SIZE, true);
    }

    pub unsafe fn on_set_bits_from<F>(&self, start: usize, mut action: F)
    where
        F: FnMut(usize),
    {
        let start_chunk = start / Self::CHUNK_SIZE;
        let bits = self.bit_array.as_slice();
        let filter_bits = self.filter.as_slice();
        let mut filter_word_start = start_chunk & !63;
        while filter_word_start < self.filter.len() {
            let mut active_chunks = load_padded_le_u64(&filter_bits[filter_word_start / 8..]);
            if filter_word_start < start_chunk {
                active_chunks &= u64::MAX << (start_chunk - filter_word_start);
            }
            let remaining = self.filter.len() - filter_word_start;
            if remaining < 64 {
                active_chunks &= (1u64 << remaining) - 1;
            }
            while active_chunks != 0 {
                let chunk_idx = filter_word_start + active_chunks.trailing_zeros() as usize;
                let start = chunk_idx * Self::CHUNK_SIZE
                    + if chunk_idx == start_chunk {
                        start % Self::CHUNK_SIZE
                    } else {
                        0
                    };
                let end = min((chunk_idx + 1) * Self::CHUNK_SIZE, self.bit_array.len());
                let mut word_start = start & !63;
                while word_start < end {
                    let mut word = load_padded_le_u64(&bits[word_start / 8..]);
                    if word_start < start {
                        word &= u64::MAX << (start - word_start);
                    }
                    if end - word_start < 64 {
                        word &= (1u64 << (end - word_start)) - 1;
                    }
                    while word != 0 {
                        action(word_start + word.trailing_zeros() as usize);
                        word &= word - 1;
                    }
                    word_start += 64;
                }
                active_chunks &= active_chunks - 1;
            }
            filter_word_start += 64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::FilteredBitArray;

    #[test]
    fn set_bit_scan_matches_reference_across_word_and_chunk_boundaries() {
        for len in [0, 1, 7, 8, 63, 64, 65, 1023, 1024, 1025, 2049] {
            for pattern in 0..4 {
                let marked: Vec<bool> = (0..len)
                    .map(|i| match pattern {
                        0 => i % 73 == 0 || i + 1 == len,
                        1 => i % 2 == 0,
                        2 => true,
                        _ => false,
                    })
                    .collect();
                let mut bits = FilteredBitArray::from_len_zeroed(len);
                for (i, is_set) in marked.iter().enumerate() {
                    if *is_set {
                        unsafe { bits.set_bit_unchecked(i) };
                    }
                }
                for start in [0, 1, 63, 64, 65, 1023, 1024, len / 2, len, len + 1] {
                    let expected: Vec<usize> = marked
                        .iter()
                        .enumerate()
                        .filter_map(|(i, is_set)| (i >= start && *is_set).then_some(i))
                        .collect();
                    let mut actual = Vec::new();
                    unsafe { bits.on_set_bits_from(start, |i| actual.push(i)) };
                    assert_eq!(
                        actual, expected,
                        "len={len}, pattern={pattern}, start={start}"
                    );
                }
            }
        }
    }

    #[test]
    fn filter_scan_matches_reference_across_filter_word_boundaries() {
        let chunk = FilteredBitArray::CHUNK_SIZE;
        let len = 130 * chunk + 9;
        let marked = [
            0,
            63,
            chunk - 1,
            chunk,
            63 * chunk + 17,
            64 * chunk,
            64 * chunk + 5,
            65 * chunk - 1,
            127 * chunk + 22,
            128 * chunk,
            len - 1,
        ];
        let mut bits = FilteredBitArray::from_len_zeroed(len);
        for index in marked {
            unsafe { bits.set_bit_unchecked(index) };
        }
        for start in [
            0,
            1,
            chunk,
            63 * chunk + 18,
            64 * chunk,
            64 * chunk + 1,
            65 * chunk,
            128 * chunk + 1,
            len,
            len + 1,
        ] {
            let expected: Vec<usize> = marked.into_iter().filter(|index| *index >= start).collect();
            let mut actual = Vec::new();
            unsafe { bits.on_set_bits_from(start, |index| actual.push(index)) };
            assert_eq!(actual, expected, "start={start}");
        }
    }

    #[test]
    fn filter_scan_matches_reference_with_every_chunk_marked() {
        let chunk = FilteredBitArray::CHUNK_SIZE;
        let len = 130 * chunk + 5;
        let marked: Vec<usize> = (0..=130).map(|i| i * chunk).collect();
        let mut bits = FilteredBitArray::from_len_zeroed(len);
        for &index in &marked {
            unsafe { bits.set_bit_unchecked(index) };
        }
        for start in [
            0,
            1,
            63 * chunk,
            63 * chunk + 1,
            64 * chunk,
            64 * chunk + 1,
            127 * chunk,
            128 * chunk + 1,
            len,
            len + 1,
        ] {
            let expected: Vec<usize> = marked
                .iter()
                .copied()
                .filter(|index| *index >= start)
                .collect();
            let mut actual = Vec::new();
            unsafe { bits.on_set_bits_from(start, |index| actual.push(index)) };
            assert_eq!(actual, expected, "start={start}");
        }
    }
}
