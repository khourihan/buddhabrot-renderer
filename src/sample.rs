use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rand::{thread_rng, Rng};
use std::{
    sync::{Arc, Mutex},
    thread,
};

use crate::{
    color::{Color, ColorChannel},
    complex::Complex,
    images::Image,
};

pub fn sample<T: Color + Clone + Copy + Send + Sync + 'static>(
    im: Arc<Mutex<Image<T>>>,
    n: u32,
    m: u32,
    progress_update: usize,
    scale: f32,
    center: Complex<f32>,
) {
    let cpus = num_cpus::get();
    let size = im.lock().unwrap().size;
    let width = im.lock().unwrap().width;
    let height = size / width;
    let iters = size * m as usize;
    let thread_progress_up = progress_update / cpus;

    let multiprogress = MultiProgress::new();
    let style = ProgressStyle::with_template("{spinner:.green} [{elapsed}] [{bar:50.white/blue}] {pos}/{len} ({eta})")
        .unwrap()
        .progress_chars("=> ")
        .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏");
    let bar = multiprogress.add(ProgressBar::new(iters as u64).with_style(style));
    bar.inc(0);

    let mut threads = Vec::new();

    for id in 0..cpus {
        // Increment the Arc's reference count and move into each thread
        let bar = bar.clone();
        let im = im.clone();

        threads.push(thread::spawn(move || {
            let mut rng = thread_rng();
            let thread_progress_offset = id * thread_progress_up;
            // Create a new thread-local image to prevent blocking
            let mut subim = Image::<T>::new(size, width);

            for i in 0..iters.div_ceil(cpus) {
                // Generate a random complex number
                let r1 = rng.gen::<f32>() * 4.0 - 2.0;
                let r2 = rng.gen::<f32>() * 4.0 - 2.0;

                // Transform random complex number into the specified frame
                let c = Complex::new(r1, r2) * scale + center;

                // Calculate the path of this complex number over n iterations
                let trajectory = mandelbrot(c, n);

                // Iterate through each point in the complex number's journey
                for z in trajectory {
                    // Convert the complex number to pixel coordinates
                    let p = (z - center) / scale * 0.25 + 0.5;
                    let px = (p.re * width as f32) as i32;
                    let py = (p.im * height as f32) as i32;

                    // Ensure the complex number is inside the image
                    if px < 0 || py < 0 || px >= width as i32 || py >= height as i32 {
                        continue;
                    }

                    // Plot the pixel
                    subim.add((px as usize, py as usize), T::one(ColorChannel::Red));
                }

                // Update the progress bar if needed
                if i != 0 && (i + thread_progress_offset) % progress_update == 0 {
                    bar.inc(progress_update as u64)
                }
            }

            // Get a mutable reference to the main image, adding the thread-local image to it
            let mut global_im = im.lock().unwrap();
            for (x, y, px) in subim.into_enumerate_pixels() {
                global_im.add((x, y), px);
            }
        }))
    }

    for thread in threads {
        let _ = thread.join();
    }

    multiprogress.clear().unwrap();
}

fn mandelbrot(c: Complex<f32>, n: u32) -> Vec<Complex<f32>> {
    let mut z_re = c.re;
    let mut z_im = c.im;

    let mut z_re_2 = z_re * z_re;
    let mut z_im_2 = z_im * z_im;

    let mut sequence = Vec::new();

    for _ in 0..n {
        sequence.push(Complex::new(z_re, z_im));

        // Update `z` via the Mandelbrot function:
        // z = z² + c
        //
        // By some algebriac simplification this reduces down to:
        // y = Im(z² + c)
        //   = Im(x² - y² + 2ixy + x₀ + iy₀)  <-- Because we only want imaginary component, we only
        //                  ^^^^        ^^^       care about terms with `i` in them.
        //   = 2xy + y₀
        //
        // x = Re(z² + c)
        //   = Re(x² - y² + 2ixy + x₀ + iy₀)  <-- Because we only want real component, we only
        //        ^^^^^^^          ^^             care about terms without `i` in them.
        //   = x² - y² + x₀
        //
        // where:
        // z = x + iy
        // z² = (x² + iy²) = x² - y² + 2ixy
        // c = x₀ + y₀
        z_im = 2.0 * z_re * z_im + c.im;
        z_re = z_re_2 - z_im_2 + c.re;

        // Update cached squares of z_re and z_im.
        z_re_2 = z_re * z_re;
        z_im_2 = z_im * z_im;

        // Compute the square of the absolute value (magnitude) of `z`.
        // This is equivalent to square of its distance from the origin.
        // This is faster than computing just the magnitude because we remove the need for a sqrt()
        // which is incredibly slow in comparison to addition and multiplication.
        // Here, the squared magnitude is computed via the pythagorean theorem, a² + b² = c²
        // where a = z_re, b = z_im, and c = z_mag.
        let z_mag_2 = z_re_2 + z_im_2;

        // If `z` escapes the set, exit.
        // Since we are now testing the square of `z_mag`, we also make sure we square the opposite
        // side of the inequality (2² = 4).
        // z_mag > 2
        // z_mag² > 2²
        if z_mag_2 > 4.0 {
            return sequence;
        }
    }

    // If the loop completes without escaping, return an empty vector
    Vec::new()
}
