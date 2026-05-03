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
    let y0:f64 = (0..n)
      .map(|i| c_lambda[i]*u.col(i)[0].re)
      .sum();
    writeln!(stationary_out,"{} {}", step, y0).unwrap();
    c_lambda.iter_mut()
      .zip(lambdas.iter())
      .for_each(|(c,lambda)| *c = *c * lambda);
      
  }
}


pub fn perturbation(temp:f64, alpha:f64, geom_theory_file:&mut File,
  graph:&Graph, steps:usize, order:usize){
  let mut geom_theory_out = BufWriter::new(geom_theory_file);
  let init_sa_matrix = graph.to_hitting_matrix(temp, 0);
  let es:Vec<Mat<f64>> = 
    (0..=steps).map(
      |i| graph.to_hitting_matrix(temp*alpha.powi(i as i32), 0) - &init_sa_matrix 
    ).collect();

  let dim = init_sa_matrix.nrows();
  debug_assert!(dim == init_sa_matrix.ncols(), "init_sa_matrix has wrong dimensions");
  let p_pows: Vec<Mat<f64>> = {
    let mut v = vec![Mat::identity(dim, dim)];
    for _ in 0..=steps { v.push(v.last().unwrap() * &init_sa_matrix); }
    v
  };
  let mut x= 0.0;
  x+=p_pows[0][(3,0)];
  writeln!(geom_theory_out, "{}", x).unwrap();
  for n in 1..steps{
    x = 0.0;
    x+=p_pows[n][(3,0)];
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
      acc + prod[(3,0)]
      });
    }
    writeln!(geom_theory_out, "{} {}", n, x).unwrap();
  }
}
