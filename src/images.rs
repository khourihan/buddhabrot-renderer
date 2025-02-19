use std::{
    slice::{Iter, IterMut},
    vec::IntoIter,
};

use crate::color::Color;

#[derive(Debug, Clone)]
pub struct Image<T: Color> {
    data: Vec<T>,
    pub size: usize,
    pub width: usize,
}

impl<T: Color + Clone + Copy> Image<T> {
    /// Creates a new, blank image.
    #[inline]
    pub fn new(size: usize, width: usize) -> Image<T> {
        Self {
            data: vec![T::empty(); size],
            size,
            width,
        }
    }

    /// Creates a new image with all pixels set to `col`.
    #[inline]
    pub fn new_fill(size: usize, width: usize, col: T) -> Image<T> {
        Self {
            data: vec![col; size],
            size,
            width,
        }
    }

    /// Gets the value of a pixel at a given `(x, y)` pixel position.
    #[inline]
    pub fn get(&self, px: (usize, usize)) -> T {
        self.data[px.1 * self.width + px.0]
    }

    /// Sets the value of a pixel at a given `(x, y)` pixel position.
    #[inline]
    pub fn set(&mut self, px: (usize, usize), col: T) {
        self.data[px.1 * self.width + px.0] = col;
    }

    /// Swaps two pixels
    #[inline]
    pub fn swap(&mut self, p1: (usize, usize), p2: (usize, usize)) {
        self.data.swap(p1.1 * self.width + p1.0, p2.1 * self.width + p2.0);
    }

    /// Adds to the value of a pixel at a given `(x, y)` pixel position.
    #[inline]
    pub fn add(&mut self, px: (usize, usize), col: T) {
        self.data[px.1 * self.width + px.0].add(col);
    }

    /// Get an iterator over every pixel in the image.
    #[inline]
    pub fn pixels(&self) -> Pixels<T> {
        Pixels { iter: self.data.iter() }
    }

    /// Get a mutable iterator over every pixel in the image.
    #[inline]
    pub fn pixels_mut(&mut self) -> PixelsMut<T> {
        PixelsMut {
            iter: self.data.iter_mut(),
        }
    }

    #[inline]
    pub fn enumerate_pixels(&self) -> EnumeratePixels<T> {
        EnumeratePixels {
            iter: self.data.iter(),
            index: 0,
            size: self.size,
            width: self.width,
        }
    }

    #[inline]
    pub fn into_enumerate_pixels(self) -> IntoEnumeratePixels<T> {
        IntoEnumeratePixels {
            iter: self.data.into_iter(),
            index: 0,
            size: self.size,
            width: self.width,
        }
    }

    #[inline]
    pub fn enumerate_pixels_mut(&mut self) -> EnumeratePixelsMut<T> {
        EnumeratePixelsMut {
            iter: self.data.iter_mut(),
            index: 0,
            size: self.size,
            width: self.width,
        }
    }
}

impl<T: Color + Clone + Copy> Default for Image<T> {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

pub struct Pixels<'a, T: Color> {
    iter: Iter<'a, T>,
}

impl<'a, T: Color> Iterator for Pixels<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

pub struct PixelsMut<'a, T: Color> {
    iter: IterMut<'a, T>,
}

impl<'a, T: Color> Iterator for PixelsMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

pub struct IntoEnumeratePixels<T: Color> {
    size: usize,
    width: usize,
    index: usize,
    iter: IntoIter<T>,
}

impl<T: Color> Iterator for IntoEnumeratePixels<T> {
    type Item = (usize, usize, T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.size {
            return None;
        }
        self.index += 1;

        Some((
            (self.index - 1) % self.width,
            (self.index - 1) / self.width,
            self.iter.next()?,
        ))
    }
}

pub struct EnumeratePixels<'a, T: Color> {
    size: usize,
    width: usize,
    index: usize,
    iter: Iter<'a, T>,
}

impl<'a, T: Color> Iterator for EnumeratePixels<'a, T> {
    type Item = (usize, usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.size {
            return None;
        }
        self.index += 1;

        Some((
            (self.index - 1) % self.width,
            (self.index - 1) / self.width,
            self.iter.next()?,
        ))
    }
}

pub struct EnumeratePixelsMut<'a, T: Color> {
    size: usize,
    width: usize,
    index: usize,
    iter: IterMut<'a, T>,
}

impl<'a, T: Color> Iterator for EnumeratePixelsMut<'a, T> {
    type Item = (usize, usize, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.size {
            return None;
        }
        self.index += 1;

        Some((
            (self.index - 1) % self.width,
            (self.index - 1) / self.width,
            self.iter.next()?,
        ))
    }
}
