use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use num;
use std::cmp;

fn l2_norm(x: Vec<f64>) -> f64 {
    return (x.iter().map(|v| v * v).sum::<f64>().sqrt());
}

fn normalize(x: Vec<f64>) -> Vec<f64> {
    let mut x_mut: Vec<f64> = x.clone();
    let norm: f64 = l2_norm(x);
    return (x_mut.iter().map(|v| v / norm).collect());
}

fn project_into_B_cr(x: Vec<f64>, c: Vec<f64>, r: f64) -> Vec<f64> {
    let x_minus_c: Vec<f64> = x.iter().zip(&c).map(|(v, c)| v - c).collect();
    let normalized_x_minus_c: Vec<f64> = normalize(x_minus_c);
    let mut projected_x: Vec<f64> = normalized_x_minus_c.iter().zip(&c).map(|(v, c)| v * r + c).collect();
    return (projected_x);
}

fn main() {
    // create test data 
    let mut data = Array::random((1000, 10), Uniform::new(0., 1.));

    // initialize parameters
    let mut c: Vec<f64> = Vec::with_capacity(10); // CC NOTE: this will be user-set parameter
    for i in 0..10 {
        c.push(0.5);
    }
    let r: f64 = 0.1; // CC NOTE: this will be user-set parameter

    // initialize beta for alpha-beta accuracy
    let beta: f64 = 0.001; // CC NOTE: this will be user-set parameter

    // get data shape
    let n: f64 = data.shape()[0] as f64;
    let d: f64 = data.shape()[1] as f64;

    // calculate clamping parameters
    let gamma_1: f64 = (d + 2. * (d * (n / beta).ln()).sqrt() + 2. * (n / beta).ln()).sqrt();
    let gamma_2: f64 = (d + 2. * (d * (1. / beta).ln()).sqrt() + 2. * (1. / beta).ln()).sqrt();
    let thresholds: Vec<f64> = vec![(r.powi(2) + 2. * r * (2. / beta).ln().sqrt() + gamma_1.powi(2)).powf(0.5),
                                    r + gamma_1];
    let clamping_threshold: f64 = thresholds.iter().fold(f64::INFINITY, |a, &b| a.min(b));

    // project data into ball
    data.genrows_mut().into_iter()
        .for_each(|mut row| row = arr1(&project_into_B_cr(row.to_vec(), c, clamping_threshold)).view());

    // clamp
    // data.gencolumns_mut().into_iter()
    //     // pair columns with values of c
    //     .zip(c.iter())
    //         // for each pairing, iterate over the cells
    //         .for_each( |(mut column, c_i)| column = column );

    // data.genrows_mut().into_iter()
    //         .for_each(|row| *row = project(row, arr1(&c).view(), r) );

    // let a: Vec<f64> = vec![1.,2.,3.];
    // let b: Vec<f64> = vec![1.,2.,3.];
    // let c: Vec<f64> = a.iter().zip(&b).map(|(a, b)| a - b).collect();
    // println!("{:?}", c);
    // println!("{:?}", l2_norm(a));

    // println!("{:?}", data);
}