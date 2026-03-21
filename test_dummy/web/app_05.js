const express = require('express');
const app = express();

app.get('/api/data', async (req, res) => {
  const items = await fetchItems(req.query);
  res.json({ success: true, data: items });
});

function fetchItems(query) {
  return new Promise(resolve => {
    setTimeout(() => resolve([1,2,3]), 100);
  });
}

app.listen(3000, () => console.log("Server running"));
