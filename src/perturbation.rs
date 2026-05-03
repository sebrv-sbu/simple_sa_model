use std::fs::File;
use std::io::{BufWriter, Write};
use std::cmp::min;
use faer::{Mat, linalg::solvers::Solve};
use faer::c64;
use crate::anneal::*;
use crate::eigentools::*;

struct Index{
  n:usize,
  curr_indices:Vec<usize>,
}

impl Index{
  fn new(n:usize, k:usize) -> Self{
    assert!(k <= n, "k must be <= n");
    let curr_indices = (0..k).collect::<Vec<usize>>();
    Index{n, curr_indices}
  }

  fn for_each(&mut self, mut f: impl FnMut(&[usize])){
    loop {
      f(&self.curr_indices);
      let k = self.curr_indices.len();
      let mut i = k-1;
      while self.curr_indices[i] == self.n - k + i {
        if i == 0 { return; }
        i -= 1;
      }
      self.curr_indices[i]+=1;
      for j in (i+1)..k{
        self.curr_indices[j] = self.curr_indices[j-1]+1;
      }
    }
  }
  fn fold<B: Copy>(&mut self, init:B, mut f:impl FnMut(B, &[usize]) -> B) -> B{
    let mut acc = init;
    self.for_each(|combo|{
      acc = f(acc, combo);
    });
    acc
  }
}

pub fn single_step_stationary_theory(c_lambda:&mut Vec<f64>, lambdas:&Vec<f64>,
  u:&Mat<c64>) -> f64{
  let n = c_lambda.len();
  let yi:f64 = (0..n)
    .map(|i| c_lambda[i] * u.col(i)[0].re)
    .sum();
  c_lambda.iter_mut()
    .zip(lambdas.iter())
    .for_each(|(c,lambda)| *c = *c * lambda);
  yi
}

pub fn coarse_grain_search(init_temp:f64, p:f64, graph:&Graph, 
  x0:&Mat<c64>)->(f64, usize){
  /* There may be a binary search method here but we don't know yet that the *
   * function has only one local minimum, so for now we do it the naive way. */
  let sa_hit_matrix = graph.to_hitting_matrix(init_temp, 0);
  let eigen = sa_hit_matrix
    .transpose() /* since we want row eigenvectors */
    .to_owned()
    .eigen()
    .unwrap();
  let u = eigen
    .U()
    .to_owned()
    .one_normalize(); /* M^T=USU^{-1} with S as eigenvalues */

  let n = x0.ncols();

  let lu = u.to_owned().partial_piv_lu();
  let y = lu.solve(&x0.transpose());
  let mut c_lambda: Vec<f64> = (0..n).map(|i| {
    debug_assert!(y[(i,0)].im.abs() < 1e-8, "unexpected imaginary component: {}", 
      y[(i,0)].im);
    y[(i,0)].re
  }).collect();
  let mut upper_bound = 0;

  let lambdas:Vec<f64> =  (0..n)
    .map(|i| { 
      debug_assert!(eigen.S()[i].im.abs() < 1e-8); 
      eigen.S()[i].re
    })
  .collect();
  let x0_float = Mat::from_fn(x0.nrows(), x0.ncols(), |i,j| x0[(i,j)].re);
  
  while single_step_stationary_theory(&mut c_lambda, &lambdas, &u) < p
  { upper_bound += 1; }
  for k in 0..=upper_bound{
    for alpha_100 in 0..100{
      let alpha = alpha_100 as f64/100.0;
      let exact_p:f64 = (&x0_float * (0..k)
        .map(|i| graph.to_hitting_matrix(init_temp * alpha.powi(i as i32), 0))
        .fold(Mat::identity(n, n), |total_soln, mat_i|{
          total_soln * mat_i
        }))[(0,0)];
      if exact_p >= p{
        return (alpha, k)
        }
      }
    }
  (1.0, upper_bound)
}
pub fn stationary_theory(temp:f64, stationary_theory_file:&mut File,
  graph:&Graph, x0:&Mat<c64>, steps:usize){

  let sa_hit_matrix = graph.to_hitting_matrix(temp, 0);
  let eigen = sa_hit_matrix
    .transpose() /* since we want row eigenvectors */
    .to_owned()
    .eigen()
    .unwrap();
  let u = eigen
    .U()
    .to_owned()
    .one_normalize(); /* M^T=USU^{-1} with S as eigenvalues */

  let n = x0.ncols();

  let lu = u.to_owned().partial_piv_lu();
  let y = lu.solve(&x0.transpose());
  let c: Vec<f64> = (0..n).map(|i| {
    debug_assert!(y[(i,0)].im.abs() < 1e-8, "unexpected imaginary component: {}", 
      y[(i,0)].im);
    y[(i,0)].re
  }).collect();
  let mut c_lambda = c.clone();
  
  let lambdas:Vec<f64> =  (0..n)
    .map(|i| { 
      debug_assert!(eigen.S()[i].im.abs() < 1e-8); 
      eigen.S()[i].re
    })
    .collect();
  
  let mut stationary_out = BufWriter::new(stationary_theory_file);
  for step in 0..steps{
    let yi = single_step_stationary_theory(&mut c_lambda, &lambdas, &u);
    writeln!(stationary_out,"{} {}", step, yi).unwrap();
  }
}

pub fn perturbation(temp:f64, alpha:f64, geom_theory_file:&mut File,
  graph:&Graph, x0_complex:&Mat<c64>, steps:usize, order:usize){
  let mut geom_theory_out = BufWriter::new(geom_theory_file);
  let init_sa_matrix = graph.to_hitting_matrix(temp, 0);
  let es:Vec<Mat<f64>> = 
    (0..=steps).map(
      |i| graph.to_hitting_matrix(temp*alpha.powi(i as i32), 0) - &init_sa_matrix 
    ).collect();
  let x0: Mat<f64> = Mat::from_fn(x0_complex.nrows(), x0_complex.ncols(),
    |i, j| x0_complex[(i, j)].re);
  let dim = init_sa_matrix.nrows();
  debug_assert!(dim == init_sa_matrix.ncols(), "init_sa_matrix has wrong dimensions");
  let p_pows: Vec<Mat<f64>> = {
    let mut v = vec![Mat::identity(dim, dim)];
    for _ in 0..=steps { v.push(v.last().unwrap() * &init_sa_matrix); }
    v
  };
  let mut x= 0.0;
  x+=(&x0 * &p_pows[0])[(0,0)];
  writeln!(geom_theory_out, "0 {}", x).unwrap();
  for n in 1..steps{
    x = 0.0;
    x+=(&x0 * &p_pows[n])[(0,0)];
    for k in 1..=min(order, n){
      let mut index = Index::new(n, k);
      x+=index.fold(0.0, |acc, combo| {
        let mut prod:Mat<f64> = p_pows[combo[0]].clone();
        for i in 0..combo.len(){
          prod = prod * &es[combo[i]];
          let gap = if i+1 < combo.len() {
            combo[i+1] - combo[i] -1
          } else { n - 1 - combo[i] };
          prod = prod * &p_pows[gap];
        }
      acc + (&x0 *&prod)[(0,0)]
      });
    }
    writeln!(geom_theory_out, "{} {}", n, x).unwrap();
  }
}
