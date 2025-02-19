use std::ops::{Add, Div, Mul, Sub};

#[derive(Clone, Copy, Debug)]
pub struct Complex<T> {
    pub re: T,
    pub im: T,
}

impl<T> Complex<T> {
    /// Creates a new instance of `Complex` with the given real and imaginary parts.
    #[inline]
    pub fn new(re: T, im: T) -> Complex<T> {
        Self { re, im }
    }

    /// Maps a function over the real and imaginary parts of a complex number.
    #[inline]
    pub fn map<F: Fn(T) -> U, U>(self, f: F) -> Complex<U> {
        Complex::<U> {
            re: f(self.re),
            im: f(self.im),
        }
    }

    /// Zips two complex numbers together.
    #[inline]
    pub fn zip<U>(self, rhs: Complex<U>) -> Complex<(T, U)> {
        Complex::<(T, U)> {
            re: (self.re, rhs.re),
            im: (self.im, rhs.im),
        }
    }
}

impl<T: Default> Default for Complex<T> {
    fn default() -> Self {
        Self {
            re: Default::default(),
            im: Default::default(),
        }
    }
}

impl<T: Add<T, Output = T>> Add for Complex<T> {
    type Output = Self;

    /// Adds two complex numbers together.
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl<T: Sub<T, Output = T>> Sub for Complex<T> {
    type Output = Self;

    /// Subtracts one complex number from another.
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl<T: Clone + Copy + Mul<T, Output = T> + Add<T, Output = T> + Sub<T, Output = T>> Mul for Complex<T> {
    type Output = Self;

    /// Computes the product of two complex numbers.
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl<T: Clone + Copy + Div<T, Output = T> + Add<T, Output = T> + Sub<T, Output = T> + Mul<T, Output = T>> Div
    for Complex<T>
{
    type Output = Self;

    /// Performs complex division on two complex numbers.
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        Self {
            re: (self.re * rhs.re + self.im * rhs.im) / (rhs.re * rhs.re + rhs.im * rhs.im),
            im: (self.im * rhs.re - self.re * rhs.im) / (rhs.re * rhs.re + rhs.im * rhs.im),
        }
    }
}

impl<T> From<(T, T)> for Complex<T> {
    fn from(value: (T, T)) -> Self {
        Complex::new(value.0, value.1)
    }
}

impl<T> From<Complex<T>> for (T, T) {
    fn from(value: Complex<T>) -> Self {
        (value.re, value.im)
    }
}

impl<T: Hypot> Complex<T> {
    /// Computes the absolute value (magnitude) of a complex number.
    #[inline]
    pub fn abs(self) -> T {
        self.re.hypotenuse(self.im)
    }
}

pub trait Hypot {
    fn hypotenuse(self, rhs: Self) -> Self;
}

impl Hypot for f32 {
    fn hypotenuse(self, rhs: Self) -> Self {
        f32::hypot(self, rhs)
    }
}

impl Hypot for f64 {
    fn hypotenuse(self, rhs: Self) -> Self {
        f64::hypot(self, rhs)
    }
}

macro_rules! impl_op_real {
    ($($op:tt, $fn:ident, $trait:ident);*) => {
        $(
            impl<T: Clone + Copy + $trait<T, Output = T>> $trait<T> for Complex<T> {
                type Output = Self;

                #[inline]
                fn $fn(self, rhs: T) -> Self::Output {
                    Self {
                        re: self.re $op rhs,
                        im: self.im $op rhs,
                    }
                }
            }
        )*
    }
}

impl_op_real!(+, add, Add; -, sub, Sub; *, mul, Mul; /, div, Div);
