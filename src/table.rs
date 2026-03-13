use anyhow::{ensure, Result};
use get_size::GetSize;

/// Periodic, equidistant lookup table that emulates wrap-around by padding one
/// extra bin on each side.
///
/// The table stores values on a uniform grid over `[min, max]` with a given
/// `step` size.  Two extra "ghost" bins are prepended and appended so that
/// nearest-neighbour lookup near the boundaries returns the periodically
/// wrapped value without any explicit modular arithmetic.
///
/// # Padding invariant
///
/// When a value is written via [`set`](Self::set):
/// - setting the *first* interior bin (`index == 1`) also copies the value
///   into the *last* ghost bin (`index == n − 1`), and
/// - setting the *last* interior bin (`index == n − 2`) also copies the value
///   into the *first* ghost bin (`index == 0`).
///
/// Writing directly into a ghost bin is rejected with an error.
///
/// # Example
///
/// ```
/// use icotable::PaddedTable;
///
/// let mut t = PaddedTable::new(0.0, 1.0, 0.5, 0.0);
/// // Interior keys: 0.0, 0.5
/// t.set(0.0, 10.0).unwrap();
/// t.set(0.5, 20.0).unwrap();
///
/// // Ghost bins mirror the boundary values:
/// assert_eq!(*t.get(-0.5).unwrap(), 20.0); // lower ghost == last interior
/// assert_eq!(*t.get(1.0).unwrap(), 10.0);  // upper ghost == first interior
/// ```
#[derive(Debug, Clone, GetSize)]
pub struct PaddedTable<T: Clone + GetSize> {
    min: f64,
    max: f64,
    res: f64,
    data: Vec<T>,
}

impl<T: Clone + GetSize> PaddedTable<T> {
    /// Create a new table spanning `[min, max]` with the given `step` size.
    ///
    /// All bins (including the two ghost bins) are initialised to
    /// `initial_value`.  The total number of bins is
    /// `⌊(max − min) / step⌋ + 3` (interior + 2 ghost).
    ///
    /// # Panics
    ///
    /// Panics if `min >= max` or `step <= 0`.
    pub fn new(min: f64, max: f64, step: f64, initial_value: T) -> Self {
        assert!(min < max && step > 0.0);
        let n = (2.0f64.mul_add(step, max - min) / step + 0.5) as usize;
        Self {
            min: min - step,
            max: max + step,
            res: step,
            data: vec![initial_value; n],
        }
    }

    /// Iterate over `(key, &value)` pairs for **all** bins, including the two
    /// ghost bins at the boundaries.
    pub fn iter(&self) -> impl Iterator<Item = (f64, &T)> {
        let min = self.min;
        let res = self.res;
        self.data
            .iter()
            .enumerate()
            .map(move |(i, value)| ((i as f64).mul_add(res, min), value))
    }

    /// Iterate over `(key, &mut value)` pairs for **all** bins, including the
    /// two ghost bins at the boundaries.
    ///
    /// # Warning
    ///
    /// Mutating a ghost bin directly will break the padding invariant.
    /// Prefer [`set`](Self::set) when writing individual values.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (f64, &mut T)> {
        let min = self.min;
        let res = self.res;
        self.data
            .iter_mut()
            .enumerate()
            .map(move |(i, value)| ((i as f64).mul_add(res, min), value))
    }

    /// Smallest non-padded key.
    pub fn min_key(&self) -> f64 {
        self.min + self.res
    }

    /// Largest non-padded key.
    pub const fn max_key(&self) -> f64 {
        self.max - self.res
    }

    /// Step size between keys.
    pub const fn key_step(&self) -> f64 {
        self.res
    }

    /// Convert a key to its nearest bin index (including ghost bins).
    ///
    /// Returns an error if `key` falls outside the padded range
    /// `[min − step, max + step]`.
    pub fn to_index(&self, key: f64) -> Result<usize> {
        let raw = (key - self.min) / self.res + 0.5;
        ensure!(raw >= 0.0, "Key {key} is below the padded range");
        let index = raw as usize;
        ensure!(index < self.data.len(), "Key {key} is above the padded range");
        Ok(index)
    }

    /// Set the value at `key`, maintaining the [padding invariant](Self).
    ///
    /// Setting a ghost bin directly is an error.  Setting the first or last
    /// interior bin automatically mirrors the value into the opposite ghost
    /// bin.
    pub fn set(&mut self, key: f64, value: T) -> Result<()> {
        let n = self.data.len();
        let index = self.to_index(key)?;

        if index == 0 || index == n - 1 {
            anyhow::bail!("Cannot set value in padded region")
        } else if index == 1 {
            self.data[n - 1] = value.clone();
        } else if index == n - 2 {
            self.data[0] = value.clone();
        }

        self.data[index] = value;
        Ok(())
    }

    /// Get a reference to the value at `key`.
    pub fn get(&self, key: f64) -> Result<&T> {
        let index = self.to_index(key)?;
        Ok(&self.data[index])
    }

    /// Get a mutable reference to the value at `key`.
    ///
    /// # Warning
    ///
    /// Mutating a boundary-adjacent value through this reference will **not**
    /// update the corresponding ghost bin.  Prefer [`set`](Self::set) when the
    /// padding invariant must be preserved.
    pub fn get_mut(&mut self, key: f64) -> Result<&mut T> {
        let index = self.to_index(key)?;
        Ok(&mut self.data[index])
    }

    /// Total number of bins, including the two ghost bins.
    pub const fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the table has no entries.
    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_table() {
        let dx = 0.1;
        let mut table = PaddedTable::new(0.0, 1.0, dx, 0.0);
        let n = table.len();
        assert_eq!(n, 12);
        for i in 0..10 {
            table.set(i as f64 * 0.1, (i + 1) as f64).unwrap();
        }
        assert_eq!(*table.get(0.0).unwrap(), 1.0);
        assert_eq!(*table.get(0.9).unwrap(), 10.0);
        assert!(table.set(0.0 - dx, 0.0).is_err());
        assert!(table.set(1.0, 0.0).is_err());
        assert_eq!(*table.get(0.0 - dx).unwrap(), 10.0); // lower padding
        assert_eq!(*table.get(1.0).unwrap(), 1.0); // upper padding
    }

    #[test]
    fn test_table_angles() {
        let res = 0.1;
        let mut table = PaddedTable::<usize>::new(0.0, 2.0 * PI, res, 0);
        // Interior size = number of bins that fit in [0, 2π)
        let n_interior = table.len() - 2;
        for i in 0..n_interior {
            table.set(i as f64 * res, i).unwrap();
        }
        assert_eq!(*table.get(0.0).unwrap(), 0);
        assert_eq!(*table.get(2.0 * PI).unwrap(), 0); // upper padding
        assert_eq!(*table.get(-res).unwrap(), n_interior - 1); // lower padding
    }
}
