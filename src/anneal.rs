use faer::{Mat};
use rand::RngExt;
use std::fs::File;
use std::io::{BufWriter, Write, BufReader, BufRead};
use std::path::Path;


struct Edge{
  neighbour: usize,
  weight: f64
}

struct Node{
  cost: f64,
  edges: Vec<Edge>,
  sum_weights: f64
}

pub struct Graph{
  nodes: Vec<Node>
}

impl Graph{
  fn new() -> Self{
    Graph { nodes: Vec::new() }
  }
  fn add_node(&mut self, cost:f64){
    self.nodes.push(Node { cost: cost, edges: Vec::new(), sum_weights: 0.0 });
  }
  fn add_edge(&mut self, from:usize, to:usize, weight:f64){
    self.nodes[from]
      .edges
      .push(Edge { neighbour: to, weight: weight } );
    self.nodes[from].sum_weights += weight;
    self.nodes[to]
      .edges
      .push(Edge { neighbour: from, weight: weight } );
    self.nodes[to].sum_weights += weight;
  }
  pub fn from_matrix(adjacency_cost_mat:&Mat<f64>) -> Self {
    let mut g = Graph::new();
    for i in 0..adjacency_cost_mat.nrows() { 
      g.add_node(adjacency_cost_mat[(i,i)]);
    }
    
    for i in 0..adjacency_cost_mat.nrows(){
      for j in i+1..adjacency_cost_mat.ncols(){
        if adjacency_cost_mat[(i,j)] >1e-8 {
          g.add_edge(i,j,adjacency_cost_mat[(i,j)]);
        }
      }
    }
    g
  }
  fn rand_neighbour(&self, id:usize) -> usize{
    let node = &self.nodes[id];
    let sum_weights = node.sum_weights;
    let bound:f64 = rand::rng().random_range(0.0..sum_weights);
    let mut mass: f64 = 0.0;
    let mut i: usize  = 0;
    while mass < bound {
      mass+=node.edges[i].weight;
      i = i + 1;
    }
    i = i - 1;
    self.nodes[id].edges[i].neighbour
  }

  fn to_matrix(&self, temp:f64)->Mat<f64>{
    let n = self.nodes.len();
    let mut sa_matrix = Mat::<f64>::zeros(n,n);
    for i in 0..n{
      let node = &self.nodes[i];
      let node_cost = node.cost;
      let sum_weights = node.sum_weights;
      let mut self_loop = 1.0;
      for edge in &node.edges{
        let neighbour = edge.neighbour;
        let weight = edge.weight;
        let neighbour_cost = self.nodes[neighbour].cost;
        sa_matrix[(i,neighbour)] = if neighbour_cost < node_cost { 
          weight / sum_weights
        } else { 
          (weight/sum_weights)*(-(neighbour_cost - node_cost)/temp).exp()
        };
        self_loop -= sa_matrix[(i,neighbour)];
      }
      sa_matrix[(i,i)] = self_loop;
    }
  sa_matrix
  }
  pub fn to_hitting_matrix(&self, temp:f64, state:usize)->Mat<f64>{
    let n = self.nodes.len();
    let mut e_i = Mat::<f64>::zeros(n,1);
    e_i[(state,0)]=1.0;
    let identity = Mat::<f64>::identity(n,n);
    let r = &e_i * e_i.transpose();
    let p = self.to_matrix(temp);
    &p+&r*(&identity-&p)
  }
}

fn anneal(graph:&Graph, i:usize, temp:f64) -> usize{
  let j:usize = graph.rand_neighbour(i);
  let cost_j:f64=graph.nodes[j].cost;
  let cost_i:f64=graph.nodes[i].cost;

  if cost_i < cost_j {
    let bound:f64 = rand::rng()
      .random_range(0.0..1.0);
    if (-(cost_j-cost_i)/temp).exp()>bound
    { j } else { i }
  } else
  { j }
}

fn geom_anneal(graph:&Graph, start_point:usize, start_temp:f64, 
  alpha:f64) -> Option<usize>{
  let mut x_k:usize = start_point;
  let mut temp:f64 = start_temp;
  let mut steps:usize = 0;
  while x_k != 0 {
    x_k=anneal(graph, x_k, temp);
    temp = temp * alpha;
    steps = steps + 1;
    if temp < 1e-8{
      return None;
    }
  }
  Some(steps)
}

fn stationary_anneal(graph:&Graph, start_point:usize, temp:f64) -> usize {
  let mut x_k:usize = start_point;
  let mut steps:usize = 0;
  while x_k != 0 {
    x_k=anneal(graph, x_k, temp);
    steps = steps + 1;
  }
  steps
}

pub fn gather_data(graph:&Graph, start_vec:Vec<f64>,
  temp:f64, alpha:f64, trials:usize, geom_file:&mut File, stationary_file:&mut File)
{
  let mut geom_out = BufWriter::new(geom_file);
  let mut stationary_out = BufWriter::new(stationary_file);
  let sum: f64 = start_vec.iter().sum();
  for _ in 0..trials{
    let mut mass:f64=0.0;
    let mut i:usize = 0;
    let bound:f64 =rand::rng().random_range(0.0..sum);
    while mass < bound{
      mass += start_vec[i];
      i = i + 1;
    }
    let start_point = i-1;
    let geom_hit = geom_anneal(&graph, start_point, temp, alpha);
    let stationary_hit = stationary_anneal(&graph, start_point, temp);
    match geom_hit {
      Some(steps) => writeln!(geom_out, "{}", steps).unwrap(),
      None => writeln!(geom_out, "NA").unwrap()
    }
    writeln!(stationary_out, "{}", stationary_hit).unwrap();
  }
}

pub fn from_file(path: impl AsRef<Path>) -> (Graph, f64, Vec<f64>){
  let file = File::open(path).unwrap();
  let reader = BufReader::new(file);
  let mut lines = reader.lines();
  let mut in_section = false;
  /* n_points, temp */
  for line in lines.by_ref() {
    let line = line.unwrap();
    if line.starts_with("$n_points_temp"){ 
      in_section = true;
      break ; 
    }
  }
  assert!(in_section, "Could not find number of points and temperature");
  let line = lines.by_ref()
    .next()
    .expect("Could not find number of points and temperature")
    .expect("error reading line");
  let mut parser = line.split_whitespace();
  let n:usize = parser
    .next().unwrap()
    .parse().unwrap();
  let temp:f64 = parser
    .next().unwrap()
    .parse().unwrap();

  let graph:Graph;
  let mut using_score_matrix = false;
  in_section = false;
  for line in lines.by_ref(){
    let line = line.unwrap();
    if line.starts_with("$begin_score_matrix") {
      using_score_matrix = true;
      in_section = true;
      break;
    } else if line.starts_with("$begin_edge_weights"){
      in_section = true;
      break;
    }
  }
  assert!(in_section, "Could not find score_matrix or edge_weights section");
  if using_score_matrix{
    let rows: Vec<Vec<f64>> = lines.by_ref()
      .take(n)
      .map(|line| {
        let line = line.expect("Error reading line");
        let row:Vec<f64>= line.split_whitespace()
          .map(|val| val.parse::<f64>().expect("Error Parsing Value"))
          .collect();
        assert!(row.len() == n, "Expected {} cols in score_matrix, only found {}", 
          n, row.len());
        row
      })
      .collect();
    assert!(rows.len() == n, "Expected {} rows in score_matrix, only found {}",
     n, rows.len()); 
    graph = Graph::from_matrix(&Mat::from_fn(n,n, |i,j| rows[i][j]));
  }
  else {
    let mut adjacency_cost_mat: Mat<f64> = Mat::zeros(n, n);
    let edges: Vec<(usize, usize, f64)> = lines.by_ref()
      .take_while(|line| {
        match line {
          Ok(s) => !s.trim().is_empty() && !s.starts_with("$"),
          Err(_) => false
        }
      })
      .map(|line| {
        let line = line.expect("Error reading line");
        let mut fields = line.split_whitespace();
        let from:usize = fields.next()
          .expect("Error: missing  'from' field")
          .parse()
          .expect("Error: 'from' field is not an integer");
        let to:usize = fields.next()
          .expect("Error: missing 'to' field")
          .parse()
          .expect("Error: 'to' field is not an integer");
        assert!(to < n && from < n, "Error: Nodes out of range {} {}",to, from);
        let weight:f64 = fields.next()
          .expect("Error: missing 'weight' field")
          .parse()
          .expect("Error: 'weight' field is not a float");
        (from, to, weight)
      })
      .collect();
    for (from, to, weight) in edges{
      assert!(adjacency_cost_mat[(from, to)] == 0.0, "Error: Assigning 
        multiple weights to {} {} pair", from, to);
      adjacency_cost_mat[(from, to)] = weight;
    }
    graph = Graph::from_matrix(&adjacency_cost_mat);
  }
  let mut vector = false;
  in_section=false;
  for line in lines.by_ref(){
    let line = line.unwrap();
    if line.starts_with("$initial_state") {
      in_section = true;
      if line.starts_with("$initial_state_vec"){
      vector = true;
      }
    break;
    }
  }

  assert!(in_section, "Could not find initial state");

  let x:Vec<f64> = if vector{
    lines.by_ref()
      .take(n)
      .map(|line| {
        line.expect("Error reading line")
          .trim()
          .parse::<f64>()
          .expect("Error parsing value")
      })
      .collect()
    } else {
      let start:usize = lines.by_ref()
        .next()
        .expect("Error: missing start index")
        .expect("Error reading line")
        .trim()
        .parse()
        .expect("Error parsing start index");
      let mut x =  vec![0.0; n];
      x[start] = 1.0;
      x
  };
  (graph, temp, x)
}
