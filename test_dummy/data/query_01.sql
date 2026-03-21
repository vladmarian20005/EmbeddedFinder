CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  created_at TIMESTAMP DEFAULT NOW(),
  status VARCHAR(50) DEFAULT 'active'
);

SELECT * FROM users WHERE status = 'active' ORDER BY created_at DESC LIMIT 100;

INSERT INTO users (name) VALUES ('test_entry');
