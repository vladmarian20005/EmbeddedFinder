CREATE TABLE orders (
  id SERIAL PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  created_at TIMESTAMP DEFAULT NOW(),
  status VARCHAR(50) DEFAULT 'active'
);

SELECT * FROM orders WHERE status = 'active' ORDER BY created_at DESC LIMIT 100;

INSERT INTO orders (name) VALUES ('test_entry');
