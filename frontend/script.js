async function askQuestion() {
    const question = document.getElementById("question").value;
    const responseDiv = document.getElementById("response");
  
    responseDiv.innerText = "Thinking... 🤔";
  
    try {
      const res = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ question: question })
      });
  
      const data = await res.json();
      responseDiv.innerText = data.answer;
    } catch (err) {
      responseDiv.innerText = "Oops! Something went wrong.";
      console.error(err);
    }
  }
  