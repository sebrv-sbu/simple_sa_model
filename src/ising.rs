use rand::RngExt;
use crate::anneal::*;
use std::fs::File;
use std::io::{BufWriter, Write, BufReader, BufRead};
use std::path::Path;

#[derive(Clone, Copy, Debug)]
struct IsingEdge{
  neighbour: usize,
  weight: f64,
}

pub struct Ising{
  dim: usize,
  sizes: Vec<usize>,
  n_points: usize,
  pub spins: Vec<bool>,
  edges: Vec<IsingEdge>,
  mu:f64,
  magnetic_field:Vec<f64>,
  pub cost: f64
}

macro_rules! weight_coord_base {
  ($i:expr, $j:expr, $dim:expr) => { (2*$dim*$i + $j) };
}
 macro_rules! local_index { 
  ($rel_index:expr) => 
  {(
    if $rel_index % 2 == 0 { $rel_index + 1 } else { $rel_index - 1 }
    )}
 }

/* Coordinates are explicit. All numbers can be converted into 
 * base [abc...]. Let us take the example of a 3,2,4 rectangle.
 * Then, we convert our number into base[324]. 
 * So 11, for example, we subtract 4 to get 5, then subtract
 * 4 again to get 1, so we have 101=11.
 * If we wanted to do this in reverse, we could try the example 
 * of 212, which is 2 units, +4, +2 sets of 2 sets of 4, 
 * which is 22.
 *                                                              */

impl Ising {
  fn new(dim:usize, sizes:Vec<usize>) -> Self{
    let n_points = sizes.iter()
      .fold(1, |prod, size| prod * size);
    let cost = 0.0;
    let magnetic_field= Vec::<f64>::new();
    let mu = 0.0;
    let spins = Vec::<bool>::new();
    let edges = Vec::<IsingEdge>::new();
    Ising{ dim, sizes, n_points, cost, magnetic_field, mu, spins, edges}
  } 
  #[inline(always)]
  fn get_edge(&self, vertex:usize, edge_index: usize) -> &IsingEdge{
    &self.edges[self.edge_index(vertex, edge_index)]
  }
  #[inline(always)]
  fn edge_index(&self, vertex:usize, edge_index:usize) -> usize{
    2 * vertex * self.dim + edge_index
  }
  fn set_edges(&mut self, weights:Vec<f64>){
    let mut edges = Vec::<IsingEdge>::new();
    let weight_coord = |i: usize, j: usize| 
      weight_coord_base!(i, j, self.dim);
    for node in 0..self.n_points{
      let mut row_edges = Vec::new();
      let start = weight_coord(node, 0);
      let end = weight_coord(node, 2 * self.dim);
      let neighbours = self.neighbours(node);
      for (&neighbour, &weight) in neighbours
          .iter()
          .zip(&weights[start..end]){
            if neighbour == node {
              row_edges.push(IsingEdge { neighbour, weight });
              continue;
            }
              if let Some(_) = row_edges.iter_mut()
                .find(|e| e.neighbour == neighbour) {
                  row_edges.push(IsingEdge { neighbour: node, weight: 0.0});
              } else {
                  row_edges.push(IsingEdge{ neighbour, weight })
                }
              }
        edges.extend(row_edges);
        }
    self.edges=edges;
  }
  fn set_mu(&mut self, mu:f64){
    self.mu=mu;
  }
  fn set_magnetic_field(&mut self, magnetic_field: Vec<f64>){
    self.magnetic_field = magnetic_field;
  }
  fn to_coord(&self, node:usize) -> Vec<usize>{
    let mut node_copy = node;
    let mut coord = vec![0; self.sizes.len()];
    for i in 0..self.sizes.len(){
      coord[i] = node_copy % self.sizes[i];
      node_copy /= self.sizes[i]
    }
    coord
  }
  fn from_coord(&self, coord:&[usize]) -> usize{
    coord.iter()
      .zip(self.sizes.iter())
      .fold((0,1), |(node, base), (&pos, &dim)| {
        (node + pos * base, base * dim)
      })
      .0
  }
  fn neighbours(&self, node:usize) -> Vec<usize>{
    let mut neighbours = Vec::<usize>::new();
    let mut node_coord = self.to_coord(node);
    for i in 0..self.sizes.len(){
      if self.sizes[i] > 1{
        node_coord[i] = (node_coord[i] + 1) % self.sizes[i];
        neighbours.push(self.from_coord(&node_coord));
        node_coord[i] = (node_coord[i] + self.sizes[i]- 2) % self.sizes[i];
        neighbours.push(self.from_coord(&node_coord));
        node_coord[i] = (node_coord[i] + 1) % self.sizes[i];
      } else {
        neighbours.push(node);
        neighbours.push(node);
      }
    }
    neighbours
  }
  fn init_cost(&mut self){
    self.cost = 
      (0..self.n_points)
      .fold(0.0, |cost, node| {
        let spin_home = self.spins[node];
        let interaction = (0..2*self.dim)
          .fold(0.0,|local_weight, neighbour_idx|{
            let edge = self.get_edge(node, neighbour_idx);
            let sign = (1 - 2 * ((self.spins[edge.neighbour] == spin_home) 
                as i32))as f64;
            local_weight + (sign * edge.weight)
          });

        let magnetic = ((1 - 2 * spin_home as i32) as f64 ) 
          * self.mu
          * self.magnetic_field[node];
      
      cost + magnetic + interaction / 2.0
    });
  }
  fn init_spins_rand(&mut self){
    self.spins = (0..self.n_points)
      .map(|_| rand::rng().random_bool(0.5))
      .collect();
  }
  fn set_spins(&mut self, spins:Vec<bool>){
    self.spins = spins;
  }
  fn cost_diff(&self, node:usize) -> f64{
    let degree = 2 * self.dim;
    let old_spin = self.spins[node];

    let interaction_diff = (0..degree)
      .fold(0.0,|local_weight, neighbour_idx|{
        let edge = self.get_edge(node, neighbour_idx);
        let neighbour_spin = self.spins[edge.neighbour];
        let was_equal = (neighbour_spin == old_spin) as i32;
        local_weight + ((2 * was_equal - 1) as f64) * edge.weight
      });

    let mag_diff = self.mu * 
      ((2 * (old_spin as i32) - 1) as f64) * 
      self.magnetic_field[node];
    2.0 * (interaction_diff + mag_diff)
  }
  fn anneal(&mut self, temp:f64) {
    let node:usize = rand::rng().random_range(0..self.n_points);
    let delta = self.cost_diff(node);
    if delta < 0.0 || (-delta / temp).exp() > rand::rng()
      .random_range(0.0..1.0) {
      self.spins[node] = !self.spins[node];
      self.cost += delta;
    }  
  }
  fn geom_anneal(&mut self, start_temp:f64, alpha: f64, max_iter:usize) 
    -> (f64, usize, Vec<bool>){
    let mut temp = start_temp;
    let mut best_cost = f64::INFINITY;
    let mut best_config = self.spins.clone();
    let mut hitting_time = 0;
    self.init_cost();
    for i in 0..max_iter{
      self.anneal(temp);
      if self.cost < best_cost{
        best_cost = self.cost;
        best_config = self.spins.clone();
        hitting_time = i;
      }
      temp *= alpha;
    }
    (best_cost, hitting_time, best_config)
  }
  fn stationary_anneal(&mut self, temp:f64, max_iter:usize) 
    -> (f64, usize, Vec<bool>){
    let mut best_cost = f64::INFINITY;
    let mut best_config = self.spins.clone();
    let mut hitting_time = 0;
    self.init_cost();
    for i in 0..max_iter{
      self.anneal(temp);
      if self.cost < best_cost{
        best_cost = self.cost;
        best_config = self.spins.clone();
        hitting_time = i;
      }
    }
    (best_cost, hitting_time, best_config)
  }
  fn to_graph(&mut self) -> Graph{
    let mut graph = Graph::new();
    assert!(self.n_points <= 64, "Error: Cannot convert
      a model with more than 64 points into a graph");
    let tot_configs:u64 = 1<<self.n_points;
    for config in 0..tot_configs{
      let mut spins = Vec::<bool>::new();
      let mut mask:u64 = 1;
      for _bit in 0..self.n_points{
        spins.push(mask & config != 0);
        mask <<= 1;
      }
      mask = 1;
      self.spins = spins;
      self.init_cost();
      graph.add_node(self.cost);
      for _bit in 0..self.n_points{
        if mask & config != 0{
          graph.add_edge(
            config as usize, 
            (config ^ mask) as usize,
            1.0
            );
        } 
        mask <<= 1;
      }
    }
    graph
  }
}

pub fn from_ising_file(path: impl AsRef<Path>) -> (Ising, f64) {
  let file = File::open(path).unwrap();
  let reader = BufReader::new(file);
  let mut lines = reader.lines();
  let mut in_section = false;
  for line in lines.by_ref(){
    let line = line.unwrap();
    if line.starts_with("$temp"){
      in_section = true;
      break
    }
  }
  assert!(in_section, "Could not find temperature");
  let line = lines.by_ref()
    .next()
    .expect("Could not find dimension and sizes")
    .expect("error reading line");
  let mut parser = line.split_whitespace();
  let temp:f64 = parser
    .next().unwrap()
    .parse().unwrap();

  for line in lines.by_ref(){
    let line = line.unwrap();
    if line.starts_with("$dim_sizes"){
      in_section = true;
      break
    }
  }
  assert!(in_section, "Could not find dimension and sizes");
  let line = lines.by_ref()
    .next()
    .expect("Could not find dimension and sizes")
    .expect("error reading line");
  let mut parser = line.split_whitespace();
  let dim:usize = parser
    .next().unwrap()
    .parse().unwrap();
  assert!(dim > 0, "Dimension cannot be 0, that is nonsensical");
  let mut sizes = Vec::<usize>::new();
  for _i in 0..dim{
    let length:usize = parser
      .next().expect("Error: Dimension and Sizes mismatch")
      .parse().expect("Error parsing sizes");
    sizes.push(length);
  }
  let mut ising_instance = Ising::new(dim, sizes);
  in_section = false;
  for line in lines.by_ref(){
    let line = line.unwrap();
    if line.starts_with("$edge_weights_start") {
      in_section = true;
      break;
    }
  }
  assert!(in_section, "Could not find edge_weights");
  let mut weights = vec![0.0; 2*ising_instance.dim*ising_instance.n_points];
  /* Difficult command time! */
  for node in 0..ising_instance.n_points{
    let mut i = 0;
    for neighbour in ising_instance.neighbours(node){
      if neighbour > node {
        let line = lines.by_ref()
          .next()
          .expect("Insufficient number of weights")
          .expect("Error reading line");
        let weight:f64 = line.split_whitespace()
          .next()
          .expect("Blank weight")
          .parse()
          .expect("Error reading line");
        weights[2*ising_instance.dim*node + i] = weight;
        weights[2*ising_instance.dim*neighbour + local_index!(i)] = weight;
      }
      i += 1;
    }
  }
  ising_instance.set_edges(weights);
  in_section=false;
  for line in lines.by_ref(){
    let line = line.unwrap();
    if line.starts_with("$mu") {
      in_section = true;
      break;
    }
  }
  assert!(in_section, "Could not find magnetic moment mu");
  ising_instance.set_mu(
    lines.by_ref()
      .next()
      .expect("Could not find magnetic moment mu")
      .expect("Error reading line below mu")
      .split_whitespace()
      .next()
      .expect("Blank magnetic moment")
      .parse::<f64>()
      .expect("Error reading magnetic moment mu")
  );
  in_section=false;
  for line in lines.by_ref(){
    let line = line.unwrap();
    if line.starts_with("$external_magnetic_field") {
      in_section = true;
      break;
    }
  }
  assert!(in_section, "Could not find external magnetic field section");
  ising_instance.set_magnetic_field(
    (0..ising_instance.n_points).map(|_| 
      lines.by_ref()
        .next()
        .expect("Error: Not enough entries in the external magnetic field")
        .expect("Error reading external magnetic field entries")
        .split_whitespace()
        .next()
        .expect("Blank magnetic field entry")
        .parse::<f64>()
        .expect("Error reading magnetic field entry")
    ).collect()
  );
  (ising_instance, temp)
}

fn config_to_u64(config:Vec<bool>)->Vec<u64>{
  config.chunks(64)
    .map(|chunk| {
      chunk.iter()
        .enumerate()
        .fold(0u64, |acc, (i, &bit)|{
          if bit { 
            acc | 1 << i
          } else {
            acc
          }
        })
    })
  .collect()
}

pub fn gather_ising_data(ising: &mut Ising, temp:f64,
  alpha:f64, trials:usize, geom_file:&mut File, stationary_file:&mut File)
{
  let mut geom_out = BufWriter::new(geom_file);
  let mut stationary_out = BufWriter::new(stationary_file);
  writeln!(geom_out, "cost\thitting_time\tconfig").unwrap();
  writeln!(stationary_out, "cost\thitting_time\tconfig").unwrap();

  for _ in 0..trials{
    ising.init_spins_rand();
    let spins = ising.spins.clone();
    let (geom_cost, geom_hit, geom_config) =
      ising.geom_anneal(temp,alpha, 1000);
    ising.set_spins(spins);
    let (stationary_cost, stationary_hit, stationary_config) = 
      ising.stationary_anneal(temp, 1000);
    let geom_len = geom_config.len();
    let stationary_len = stationary_config.len();
    let geom_str: String = config_to_u64(geom_config)
      .iter()
      .flat_map(|&num| num.to_le_bytes())
      .take((geom_len + 7)/8)
      .map(|byte| format!("{:02x}", byte))
      .collect::<Vec<String>>()
      .join(" ");
    writeln!(geom_out, "{geom_cost}\t{geom_hit}\t{geom_str}").unwrap();
    
    let stationary_str: String = config_to_u64(stationary_config)
      .iter()
      .flat_map(|&num| num.to_le_bytes())
      .take((stationary_len + 7)/8)
      .map(|byte| format!("{:02x}", byte))
      .collect::<Vec<String>>()
      .join(" ");
    writeln!(stationary_out, 
      "{stationary_cost}\t{stationary_hit}\t{stationary_str}").unwrap();
  }

}     
