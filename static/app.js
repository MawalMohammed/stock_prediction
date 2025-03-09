// app.js

// Add event listener to the form
const form = document.querySelector('form');
form.addEventListener('submit', async (event) => {
  event.preventDefault(); // Prevent form submission

  const stockSymbol = document.getElementById('stock_symbol').value;

  try {
    // Send AJAX request to Flask route to get prediction
    const response = await fetch('/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: `stock_symbol=${stockSymbol}`,
    });

    if (response.ok) {
      const prediction = await response.json();
      const resultDiv = document.getElementById('prediction-result');
      resultDiv.innerHTML = `<p>Predicted Stock Price: ${prediction.price}</p>`;
    } else {
      console.error('Error fetching prediction');
    }
  } catch (error) {
    console.error('Error:', error);
  }
});