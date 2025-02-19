use clap::{error::ErrorKind, CommandFactory, Parser, Subcommand, ValueEnum};
use image::GenericImageView;
use std::{
    path::PathBuf,
    str::FromStr,
    sync::{Arc, Mutex},
};

use buddhabrot::{
    color::{Color, Float, Rgb},
    complex::Complex,
    images::Image,
    sample::sample,
};

fn normalize_im<T: Color + Clone + Copy + Send + Sync + 'static>(im: &mut Image<T>) {
    let mut max = T::empty();
    for pixel in im.pixels() {
        max = max.max(*pixel);
    }

    for pixel in im.pixels_mut() {
        pixel.cdiv_assign(max);
    }
}

fn reflect_im<T: Color + Clone + Copy>(im: &mut Image<T>) {
    for i in 0..im.size / 2 {
        let x = i % im.width;
        let y = i / im.width;
        let c1 = im.get((x, y));
        let c2 = im.get((x, im.width - y - 1));
        im.add((x, im.width - y - 1), c1);
        im.add((x, y), c2);
    }
}

fn rotate_im<T: Color + Clone + Copy>(im: &mut Image<T>) {
    for i in 0..im.size {
        let x = i % im.width;
        let y = i / im.width;
        if x > y {
            im.swap((x, y), (y, x));
        }
    }
}

fn fuse(im1: Image<f32>, im2: Image<f32>, im3: Image<f32>) -> Image<Rgb> {
    let mut im = Image::<Rgb>::new(im1.size, im1.width);
    for (x, y, px) in im1.into_enumerate_pixels() {
        let py = im2.get((x, y));
        let pz = im3.get((x, y));
        im.set((x, y), Rgb::new(px, py, pz));
    }
    im
}

fn parse_color(s: &str) -> Result<(f32, f32, f32), String> {
    let e = format!("{} is not a valid rgb color", s);
    if s.starts_with('#') {
        let v = s
            .chars()
            .skip(1)
            .enumerate()
            .flat_map(|(i, c)| {
                if i != 0 && i % 2 == 0 { Some(' ') } else { None }
                    .into_iter()
                    .chain(std::iter::once(c))
            })
            .collect::<String>();

        let mut v = v.split(' ').map(|s| {
            let bytes = u8::from_str_radix(s, 16).unwrap();
            bytes as f32 / 255.0
        });

        Ok((
            v.next().ok_or(e.clone())?,
            v.next().ok_or(e.clone())?,
            v.next().ok_or(e)?,
        ))
    } else {
        let mut v = s.split(',');
        Ok((
            v.next().ok_or(e.clone())?.parse::<f32>().map_err(|_| e.clone())?,
            v.next().ok_or(e.clone())?.parse::<f32>().map_err(|_| e.clone())?,
            v.next().ok_or(e.clone())?.parse::<f32>().map_err(|_| e)?,
        ))
    }
}

fn write_rgb(im: Image<Rgb>, mut file: PathBuf, png: bool) {
    if png {
        file.set_extension("png");
        let mut imgbuf = image::ImageBuffer::new(im.width as u32, im.width as u32);

        for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
            let c = im.get((x as usize, y as usize)).map(|x| x * 255.0);
            let v = c.to_tuple_rgb();
            *pixel = image::Rgb([v.0 as u8, v.1 as u8, v.2 as u8])
        }

        imgbuf.save(file).unwrap();
    } else {
        file.set_extension("exr");
        exr::image::write::write_rgb_file(file, im.width, im.width, |x, y| im.get((x, y)).to_tuple_rgb()).unwrap();
    }
}

fn load_image(input_file: &PathBuf) -> clap::error::Result<Image<Rgb>, clap::Error> {
    Ok(if let Some(extension) = input_file.extension() {
        if extension == "exr" {
            exr::image::read::read_first_rgba_layer_from_file(
                input_file,
                |resolution, _| Image::<Rgb>::new(resolution.width() * resolution.height(), resolution.width()),
                |image: &mut Image<Rgb>, pos: exr::math::Vec2<usize>, (r, g, b, _a): (f32, f32, f32, f32)| {
                    image.set((pos.x(), pos.y()), Rgb::new(r, g, b))
                },
            )
            .unwrap()
            .layer_data
            .channel_data
            .pixels
        } else if extension == "png" {
            let png = image::open(input_file).unwrap();
            let mut im = Image::<Rgb>::new((png.width() * png.height()) as usize, png.width() as usize);

            for (x, y, px) in im.enumerate_pixels_mut() {
                let c = png.get_pixel(x as u32, y as u32);
                *px = Rgb::new(c.0[0] as f32 / 255.0, c.0[1] as f32 / 255.0, c.0[2] as f32 / 255.0);
            }

            im
        } else {
            let err = Cli::command().error(
                ErrorKind::Io,
                format!("file {:?} is invalid; expected either exr or png file", input_file),
            );
            err.print()?;
            return Err(err);
        }
    } else {
        let err = Cli::command().error(
            ErrorKind::Io,
            format!("file {:?} is invalid; expected either exr or png file", input_file),
        );
        err.print()?;
        return Err(err);
    })
}

#[derive(Parser)]
#[command(version, author, about)]
pub struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)]
enum Commands {
    Generate {
        /// The number of mandelbrot iterations each complex number undegoes.
        n_iterations: u32,

        /// The number of times to sample the image. (num_samples = image_width * image_height *
        /// samples).
        samples: u32,

        /// The width and height of the image in pixels. Recommended to be a power of 2.
        image_size: u32,

        /// The number of color channels to write to.
        #[arg(value_enum)]
        mode: ColorChannelMode,

        #[arg(short, long, value_name = "PROGRESS_UPDATE")]
        progress_update: Option<u32>,

        /// The file to write the image to, excluding the extension.
        #[arg(short, long, value_name = "FILENAME", default_value = "buddhabrot")]
        file: PathBuf,

        /// Whether or not to overwrite the file if it already exists.
        #[arg(short, long)]
        overwrite: bool,

        #[arg(short, long, default_value = "1")]
        scale: f32,

        #[arg(short, long, value_parser = parse_complex::<f32>, default_value = "0,0")]
        center: Complex<f32>,

        /// Whether to output the image in PNG format. If false, uses EXR. Note that this
        /// automatically normalizes the image beforehand.
        #[arg(long)]
        png: bool,

        /// Whether or not to normalize all pixel values between 0-1 before writing the image.
        #[arg(long)]
        normalize: bool,

        /// Whether or not to rotate the resulting image. Useful only when rendering the full
        /// buddhabrot.
        #[arg(long)]
        rotate: bool,

        /// Whether or not to reflect the resulting image and add it back to the original. This
        /// effectively doubles the number of samples but only works when rendering a symmetrical
        /// region of the fractal.
        #[arg(long)]
        reflect: bool,
    },
    Process {
        /// The full input file path to process, including the extension.
        input_file: PathBuf,

        #[command(subcommand)]
        colorize: Option<ColorizeCommand>,

        /// The output file path, excluding the extension. When unspecified, overwrites the original file.
        #[arg(short, long, value_name = "OUTFILE")]
        file: Option<PathBuf>,

        /// The exposure of the image.
        ///
        /// Recommended value: 2.5
        #[arg(short, long, value_name = "EXPOSURE")]
        exposure: Option<f32>,

        /// The gamma of the image.
        ///
        /// Recommended value: 0.45
        #[arg(short, long, value_name = "GAMMA")]
        gamma: Option<f32>,

        /// The black point of the image, or the threshold at which anything lower gets clamped to
        /// full black.
        #[arg(short, long, value_name = "BLACK_POINT")]
        black_point: Option<f32>,

        /// Whether to output the image in PNG format. If false, uses EXR. Note that this
        /// automatically normalizes and clamps the image.
        #[arg(long)]
        png: bool,

        /// Whether or not to clamp all pixels to a value between 0-1.
        #[arg(long)]
        clamp: bool,

        /// Whether or not to normalize all pixel values between 0-1 before writing the image.
        #[arg(long)]
        normalize: bool,
    },
    Fuse {
        /// The full input file path to fuse into the red channel, including the extension.
        #[arg(short, long, value_name = "RED_CHANNEL_FILE")]
        red_file: PathBuf,

        /// The full input file path to fuse into the blue channel, including the extension.
        #[arg(short, long, value_name = "GREEN_CHANNEL_FILE")]
        green_file: Option<PathBuf>,

        /// The full input file path to fuse into the blue channel, including the extension.
        #[arg(short, long, value_name = "BLUE_CHANNEL_FILE")]
        blue_file: Option<PathBuf>,

        /// The output file path, excluding the extension. When unspecified, overwrites the original file.
        #[arg(short, long, value_name = "OUTFILE")]
        file: PathBuf,

        /// Whether or not to output the file in PNG format.
        #[arg(long)]
        png: bool,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum ColorChannelMode {
    /// Write to 1 color channel.
    R,
    /// Write to 2 color channels.
    Rg,
    /// Write to 3 color channels.
    Rgb,
}

#[derive(Subcommand)]
enum ColorizeCommand {
    /// Colorize the image with custom colors, only using values from the red color channel.
    ///
    /// Note: many EXR image viewers aren't very good at interpreting the colorized output, so
    /// it is recommended to use the --png flag when doing so.
    ColorizeR {
        #[arg(long, value_name = "MIN_RED_COLOR", value_parser = parse_color)]
        minr: (f32, f32, f32),

        #[arg(long, value_name = "MAX_RED_COLOR", value_parser = parse_color)]
        maxr: (f32, f32, f32),
    },
    /// Colorize the image with custom colors, using values from the red and green color channels.
    ///
    /// Note: many EXR image viewers aren't very good at interpreting the colorized output, so
    /// it is recommended to use the --png flag when doing so.
    ColorizeRg {
        #[arg(long, value_name = "MIN_RED_COLOR", value_parser = parse_color)]
        minr: (f32, f32, f32),

        #[arg(long, value_name = "MAX_RED_COLOR", value_parser = parse_color)]
        maxr: (f32, f32, f32),

        #[arg(long, value_name = "MIN_GREEN_COLOR", value_parser = parse_color)]
        ming: (f32, f32, f32),

        #[arg(long, value_name = "MAX_GREEN_COLOR", value_parser = parse_color)]
        maxg: (f32, f32, f32),
    },
    /// Colorize the image with custom colors, using values from the red, green, and blue color channels.
    ///
    /// Note: many EXR image viewers aren't very good at interpreting the colorized output, so
    /// it is recommended to use the --png flag when doing so.
    ColorizeRgb {
        #[arg(long, value_name = "MIN_RED_COLOR", value_parser = parse_color)]
        minr: (f32, f32, f32),

        #[arg(long, value_name = "MAX_RED_COLOR", value_parser = parse_color)]
        maxr: (f32, f32, f32),

        #[arg(long, value_name = "MIN_GREEN_COLOR", value_parser = parse_color)]
        ming: (f32, f32, f32),

        #[arg(long, value_name = "MAX_GREEN_COLOR", value_parser = parse_color)]
        maxg: (f32, f32, f32),

        #[arg(long, value_name = "MIN_BLUE_COLOR", value_parser = parse_color)]
        minb: (f32, f32, f32),

        #[arg(long, value_name = "MAX_BLUE_COLOR", value_parser = parse_color)]
        maxb: (f32, f32, f32),
    },
}

fn parse_complex<T>(s: &str) -> Result<Complex<T>, String>
where
    T: FromStr + Copy,
{
    let seps = s.chars().filter(|&c| c == ',').count();

    if seps != 1 {
        return Err(format!(
            "expected complex number to have exactly one separator but got {seps}."
        ));
    }

    let mut parts = s.split(',');

    let re: T = parts
        .next()
        .unwrap()
        .parse()
        .map_err(|_| format!("could not parse real component of complex number."))?;

    let im: T = parts
        .next()
        .unwrap()
        .parse()
        .map_err(|_| format!("could not parse imaginary component of complex number."))?;

    Ok(Complex::new(re, im))
}

fn main() -> clap::error::Result<(), clap::Error> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            n_iterations,
            samples,
            image_size,
            mode,
            progress_update,
            mut file,
            overwrite,
            scale,
            center,
            png,
            normalize,
            rotate,
            reflect,
        } => {
            let im_width = image_size as usize;
            let im_size = im_width * im_width;
            let progress_update = if let Some(up) = progress_update {
                up as usize
            } else {
                im_size * 2
            };

            file.set_extension(if png { "png" } else { "exr" });

            if file.exists() && !overwrite {
                let err = Cli::command().error(
                    ErrorKind::ValueValidation,
                    format!("file {:?} already exists. to overwrite it, use the -o flag", file),
                );
                return Ok(err.print()?);
            }

            let start_time = std::time::Instant::now();
            let mut im = match mode {
                ColorChannelMode::R => {
                    let im1 = Arc::new(Mutex::new(Image::<Float>::new(im_size, im_width)));
                    sample(im1.clone(), n_iterations, samples, progress_update, scale, center);

                    let im = Arc::try_unwrap(im1).unwrap().into_inner().unwrap();
                    fuse(im.clone(), im.clone(), im)
                },
                ColorChannelMode::Rg => {
                    let im1 = Arc::new(Mutex::new(Image::<Float>::new(im_size, im_width)));
                    sample(im1.clone(), n_iterations, samples, progress_update, scale, center);

                    let im2 = Arc::new(Mutex::new(Image::<Float>::new(im_size, im_width)));
                    sample(im2.clone(), n_iterations / 10, samples, progress_update, scale, center);

                    let im1 = Arc::try_unwrap(im1).unwrap().into_inner().unwrap();
                    let im2 = Arc::try_unwrap(im2).unwrap().into_inner().unwrap();
                    fuse(im1, im2, Image::<f32>::new(im_size, im_width))
                },
                ColorChannelMode::Rgb => {
                    let im1 = Arc::new(Mutex::new(Image::<Float>::new(im_size, im_width)));
                    sample(im1.clone(), n_iterations, samples, progress_update, scale, center);

                    let im2 = Arc::new(Mutex::new(Image::<Float>::new(im_size, im_width)));
                    sample(im2.clone(), n_iterations / 10, samples, progress_update, scale, center);

                    let im3 = Arc::new(Mutex::new(Image::<Float>::new(im_size, im_width)));
                    sample(im3.clone(), n_iterations / 100, samples, progress_update, scale, center);

                    let im1 = Arc::try_unwrap(im1).unwrap().into_inner().unwrap();
                    let im2 = Arc::try_unwrap(im2).unwrap().into_inner().unwrap();
                    let im3 = Arc::try_unwrap(im3).unwrap().into_inner().unwrap();
                    fuse(im1, im2, im3)
                },
            };
            let elapsed = start_time.elapsed();
            println!(
                "Finished rendering buddhabrot in {}.",
                humantime::format_duration(std::time::Duration::new(elapsed.as_secs(), 0))
            );

            if normalize {
                normalize_im(&mut im);
            }

            if reflect {
                reflect_im(&mut im);
            }

            if rotate {
                rotate_im(&mut im);
            }

            if file.exists() && overwrite {
                std::fs::remove_file(file.clone()).unwrap();
            }

            write_rgb(im, file, png);
        },
        Commands::Process {
            mut input_file,
            colorize,
            file,
            exposure,
            gamma,
            black_point,
            png,
            clamp,
            normalize,
        } => {
            let mut im = load_image(&input_file)?;

            if png || normalize {
                normalize_im(&mut im);
            }

            if let Some(exp) = exposure {
                for px in im.pixels_mut() {
                    px.r *= exp;
                    px.g *= exp;
                    px.b *= exp;
                }
            }

            if let Some(gam) = gamma {
                for px in im.pixels_mut() {
                    px.r = px.r.powf(1.0 / gam);
                    px.g = px.g.powf(1.0 / gam);
                    px.b = px.b.powf(1.0 / gam);
                }
            }

            if let Some(thres) = black_point {
                for px in im.pixels_mut() {
                    px.r = if px.r < thres { 0.0 } else { px.r };
                    px.g = if px.g < thres { 0.0 } else { px.g };
                    px.b = if px.b < thres { 0.0 } else { px.b };
                }
            }

            if png || clamp {
                for px in im.pixels_mut() {
                    px.r = px.r.clamp(0.0, 1.0);
                    px.g = px.g.clamp(0.0, 1.0);
                    px.b = px.b.clamp(0.0, 1.0);
                }
            }

            if let Some(color) = colorize {
                let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;

                let f = |(r, g, b): (f32, f32, f32)| match color {
                    ColorizeCommand::ColorizeR { minr, maxr } => (
                        lerp(minr.0, maxr.0, r),
                        lerp(minr.1, maxr.1, r),
                        lerp(minr.2, maxr.2, r),
                    ),
                    ColorizeCommand::ColorizeRg { minr, maxr, ming, maxg } => (
                        lerp(minr.0, maxr.0, r) * 0.5 + lerp(ming.0, maxg.0, g) * 0.5,
                        lerp(minr.1, maxr.1, r) * 0.5 + lerp(ming.1, maxg.1, g) * 0.5,
                        lerp(minr.2, maxr.2, r) * 0.5 + lerp(ming.2, maxg.2, g) * 0.5,
                    ),
                    ColorizeCommand::ColorizeRgb {
                        minr,
                        maxr,
                        ming,
                        maxg,
                        minb,
                        maxb,
                    } => (
                        lerp(minr.0, maxr.0, r) / 3.0 + lerp(ming.0, maxg.0, g) / 3.0 + lerp(minb.0, maxb.0, b) / 3.0,
                        lerp(minr.1, maxr.1, r) / 3.0 + lerp(ming.1, maxg.1, g) / 3.0 + lerp(minb.1, maxb.1, b) / 3.0,
                        lerp(minr.2, maxr.2, r) / 3.0 + lerp(ming.2, maxg.2, g) / 3.0 + lerp(minb.2, maxb.2, b) / 3.0,
                    ),
                };

                for px in im.pixels_mut() {
                    *px = f((*px).into()).into();
                }
            }

            input_file.set_extension(if png { "png" } else { "exr" });
            let out_file = if let Some(f) = &file { f } else { &input_file };
            if out_file.exists() {
                std::fs::remove_file(out_file).unwrap();
            }

            write_rgb(im, out_file.to_path_buf(), png);
        },
        Commands::Fuse {
            red_file,
            green_file,
            blue_file,
            file,
            png,
        } => {
            let red_im = load_image(&red_file)?;
            let mut im = Image::<Rgb>::new(red_im.size, red_im.width);

            for (x, y, px) in im.enumerate_pixels_mut() {
                px.r = red_im.get((x, y)).r;
            }

            if let Some(path) = green_file {
                let green_im = load_image(&path)?;

                if green_im.width != im.width || green_im.size != im.size {
                    let err = Cli::command().error(
                        ErrorKind::Io,
                        format!("file {:?} has different dimensions than {:?}", path, red_file),
                    );
                    err.print()?;
                    return Err(err);
                }

                for (x, y, px) in im.enumerate_pixels_mut() {
                    px.g = green_im.get((x, y)).r;
                }
            }

            if let Some(path) = blue_file {
                let blue_im = load_image(&path)?;

                if blue_im.width != im.width || blue_im.size != im.size {
                    let err = Cli::command().error(
                        ErrorKind::Io,
                        format!("file {:?} has different dimensions than {:?}", path, red_file),
                    );
                    err.print()?;
                    return Err(err);
                }

                for (x, y, px) in im.enumerate_pixels_mut() {
                    px.b = blue_im.get((x, y)).r;
                }
            }

            write_rgb(im, file, png);
        },
    }

    Ok(())
}
