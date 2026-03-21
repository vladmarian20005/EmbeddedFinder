use std::collections::HashMap;

fn main() {
    let mut scores: HashMap<String, i32> = HashMap::new();
    scores.insert("Alice".to_string(), 95);
    scores.insert("Bob".to_string(), 87);
    
    for (name, score) in &scores {
        println!("{}: {}", name, score);
    }
    
    let total: i32 = scores.values().sum();
    println!("Average: {}", total / scores.len() as i32);
}
