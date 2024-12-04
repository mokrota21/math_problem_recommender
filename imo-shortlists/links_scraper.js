const puppeteer = require("puppeteer");

const URL = "https://artofproblemsolving.com/community/c3223_imo_shortlist";

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function run(url) {
  let browser;
  try {
    console.log("Launching local browser...");
    // Launch the browser locally
    browser = await puppeteer.launch({
      headless: false, // Set to true if you don't need the browser UI
      defaultViewport: null, // Use full screen
      args: ['--start-maximized'], // Open the browser in maximized window
    });

    console.log("Connected! Navigating to site...");
    const page = await browser.newPage();
    
    // Enhanced navigation with error handling
    const response = await page.goto(url, { 
      waitUntil: "networkidle2", 
      timeout: 60000 
    });

    console.log("Page status:", response.status());
    
    // Wait for the container to be present
    await page.waitForSelector('.cmty-folder-grid-container', { 
      visible: true, 
      timeout: 30000 
    });

    console.log("Waiting for content to load...");
    
    await sleep(2000);

    await scrollPage(page, 20000);

    console.log("Parsing data...");
    const data = await parse(page);

    console.log(`Data parsed: ${JSON.stringify(data, null, 2)}`);
    // Save the data to JSON file
    await saveToJsonFile('scraped_data.json', data);

    return data;
  } catch (error) {
    console.error("Error in scraping process:", error);
    throw error;
  } finally {
    if (browser) {
      await browser.close();
    }
  }
}

async function scrollPage(page, duration) {
  const startTime = Date.now(); // Record the start time

  // Loop to simulate scrolling every 100ms
  while (Date.now() - startTime < duration) {
    await page.evaluate(() => {
      window.scrollBy(0, 100); // Scroll down by 10 pixels vertically
    });
    await sleep(100); // Wait for 100ms before scrolling again
  }

  console.log(`Scrolling stopped after ${duration / 1000} seconds.`);
}

async function parse(page) {
  return await page.evaluate(() => {
    // Select all divs with the specific class within the grid container
    const folderDivs = document.querySelectorAll(
      '.cmty-folder-grid-container .cmty-category-cell.cmty-category-cell-folder'
    );

    // Map through the divs to extract information
    const results = Array.from(folderDivs).map(div => {
      // Find the link element
      const linkElement = div.querySelector('a.cmty-full-cell-link');

      // Get title from the parent div's title attribute
      const title = linkElement.parentElement ? linkElement.parentElement.title : '';

      return {
        title: title || '', // Get title from div's title attribute
        url: linkElement ? linkElement.href : '', // Get URL from link
        text: linkElement ? linkElement.textContent.trim() : '' // Optional: get link text
      };
    }).filter(item => item.url); // Filter out entries without a URL

    return results;
  });
}

// Save the data to a JSON file
const fs = require('fs/promises');
async function saveToJsonFile(filename, data) {
  try {
    const jsonString = JSON.stringify(data, null, 2); // Format the JSON data
    await fs.writeFile(filename, jsonString, "utf-8"); // Write to file
    console.log(`Data successfully saved to ${filename}`);
  } catch (err) {
    console.error("Error writing to JSON file:", err);
  }
}

// Main execution
run(URL)
  .then(data => console.log("Scraping finished"))
  .catch(err => console.error("Scraping failed:", err));
