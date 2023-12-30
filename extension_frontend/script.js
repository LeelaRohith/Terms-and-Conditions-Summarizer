const scrap_data = () => {
  const promise = new Promise((resolve, reject) => {
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
      if (tabs.length > 0) {
        chrome.scripting.executeScript(
          {
            target: { tabId: tabs[0].id },
            function: function () {
              return {
                title: document.title,
                text: document.documentElement.innerText,
              };
            },
          },
          function (result) {
            if (chrome.runtime.lastError) {
              console.error(chrome.runtime.lastError);
            } else {
              document.getElementById("title").innerHTML =
                result[0].result.title;
              // document.getElementById('content').innerHTML =
              resolve(result[0].result.text);
            }
          }
        );
      } else {
        console.error("No active tabs found in the current window.");
      }
    });
  });
  return promise;
};

const getPredictedData = (scraped_data) => {
  console.log("scrap scraped_data ", scraped_data);
  fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(scraped_data),
  })
    .then(function (response) {
      if (!response.ok) throw new Error("Network response was not ok");
      return response.json();
    })
    .then((response_data) => {
      const data = response_data.data;
      console.log(data);
      getCategories(data).then(onClick);
    });
};

const getCategories = (data) => {
  const categories_list = Object.keys(data);
  const categories_div = document.getElementById("categories");

  const promise = new Promise((resolve, reject) => {
    categories_list.forEach((category) => {
      const div_element = document.createElement("div");
      //   div_element.className = 'category_item'
      div_element.classList.add("category_item");
      div_element.textContent = category;
      categories_div.appendChild(div_element);
    });

    if (categories_list.length > 0) {
      resolve(data);
    } else {
      const err = new Error("categories_list is empty");
      reject(err);
    }
  });
  return promise;
};

const onClick = (data) => {
  const content_div = document.getElementById("content");
  const categories = document.querySelectorAll(".category_item");

  console.log("onc ", data, categories);
  const title_element = document.createElement("div");
  title_element.classList.add("title_item");

  title_element.textContent = categories[0].innerHTML;
  content_div.appendChild(title_element);

  data[categories[0].innerHTML].forEach(function (content_text) {
    const content_element = document.createElement("div");
    content_element.classList.add("content_item");
    content_element.textContent = "➡️  " + content_text;
    content_div.appendChild(content_element);
  });

  categories.forEach((category) => {
    category.addEventListener("click", (event) => {
      while (content_div.firstChild) {
        content_div.removeChild(content_div.firstChild);
      }

      const title_element2 = document.createElement("div");
      title_element2.classList.add("title_item");

      title_element2.textContent = category.innerHTML;
      content_div.appendChild(title_element2);

      console.log(event.target.innerHTML);
      data[event.target.innerHTML].forEach(function (content_text) {
        const content_element = document.createElement("div");
        content_element.classList.add("content_item");
        content_element.textContent = "➡️  " + content_text;

        content_div.appendChild(content_element);
      });
    });
  });
};

scrap_data().then((data) => getPredictedData(data));
