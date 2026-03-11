use anyhow::{ensure, Result};
use get_size::GetSize;

pub type PaddedTable1D = PaddedTable<f64>;
pub type PaddedTable2D = PaddedTable<PaddedTable1D>;

/// Periodic, equidistant table that emulates periodicity by padding edges.
#[derive(Debug, Clone, GetSize)]
pub struct PaddedTable<T: Clone + GetSize> {
    min: f64,
    max: f64,
    res: f64,
    data: Vec<T>,
}

impl<T: Clone + GetSize> PaddedTable<T> {
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

    pub fn iter(&self) -> impl Iterator<Item = (f64, &T)> {
        let min = self.min;
        let res = self.res;
        self.data
            .iter()
            .enumerate()
            .map(move |(i, value)| ((i as f64).mul_add(res, min), value))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (f64, &mut T)> {
        let min = self.min;
        let res = self.res;
        self.data
            .iter_mut()
            .enumerate()
            .map(move |(i, value)| ((i as f64).mul_add(res, min), value))
    }

    pub fn min_key(&self) -> f64 {
        self.min + self.res
    }

    pub const fn max_key(&self) -> f64 {
        self.max - self.res
    }

    pub const fn key_step(&self) -> f64 {
        self.res
    }

    pub fn to_index(&self, key: f64) -> Result<usize> {
        let index = ((key - self.min) / self.res + 0.5) as usize;
        ensure!(index < self.data.len(), "Index out of range");
        Ok(index)
    }

    /// Set value with periodic padding at boundaries.
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

    pub fn get(&self, key: f64) -> Result<&T> {
        let index = self.to_index(key)?;
        Ok(&self.data[index])
    }

    pub fn get_mut(&mut self, key: f64) -> Result<&mut T> {
        let index = self.to_index(key)?;
        Ok(&mut self.data[index])
    }

    pub const fn len(&self) -> usize {
        self.data.len()
    }
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
