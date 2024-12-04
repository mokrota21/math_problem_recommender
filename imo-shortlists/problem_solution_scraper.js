const puppeteer = require('puppeteer');

async function scrapeDivs(link, className) {
  // Launch the browser and open a new page
  const browser = await puppeteer.launch();
  const page = await browser.newPage();

  // Navigate to the URL
  await page.goto(link);

  // Wait for the divs with the specified class to load
  await page.waitForSelector(`.${className}`);

  // Extract all divs with the specified class name
  const divElements = await page.$$(`.${className}`);

  // Iterate over each div element and call the processDiv function
  for (const divElement of divElements) {
    await getContentSolution(divElement);
  }

  // Close the browser
  await browser.close();
}

// async function getNComments(page, div_class, n=2) {
//     await page.waitForSelector(`.${div_class}`);
//     const comments = await page.$$(`.${div_class}`);
//     const required_comments = await comments.slice(0, n);
//     for (const comment of required_comments) {
//         // Ging line by line
//         const lines = await comment.evaluate((divs) => {
//             return divs.map(div => {
//                 // You can modify this logic to capture different kinds of text inside the div
//                 const textContent = div.innerText.trim(); // Grab the visible text
//                 return textContent.split('\n').map(line => line.trim()); // Split by lines and trim
//             });    
//         });
//         console.log(JSON.stringify(data, null, 2));
        
        // Extract the child nodes (including text nodes and images) of the div
        // const nodes = await comment.evaluate((div) => {
        //     const children = [];

        //     function traverse(node) {
        //         if (node.tagName === 'IMG') {
        //             children.push({ type: 'latex', content: node.alt });
        //         } else if (node.textContent.trim()) {
        //             children.push({ type: 'text', content: node.textContent.trim() });
        //         }
        
        //         // Recursively traverse all child nodes
        //         node.childNodes.forEach(traverse);
        //     }
        
        //     // Start traversing from the root div
        //     traverse(div);
        
        //     return children;
        // });

        // Now process the nodes, keeping order intact
        // const result = [];
        // for (const node of nodes) {
        //     if (node.type === 'text') {
        //         result.push(node.content.trim()); // Add raw text (remove extra spaces)
        //     } else if (node.type === 'latex') {
        //         result.push(`${node.content}`); // LaTeX wrapped in math mode
        //     } else {
        //         console.log(`Unknown data. Content: ${node.content}. Type: ${node.type}`);
        //     }
        // }

        // // Join the result to keep the content in order (text and LaTeX together)
        // const finalContent = result.join(' ');

        // // Log the extracted content
        // console.log(finalContent);
    // }
// }

async function getNComments(page, div_class, n = 2) {
    await page.waitForSelector(`.${div_class}`);
    const comments = await page.$$(`.${div_class}`);
    
    // Slice to get only the required number of comments
    const required_comments = comments.slice(0, n);
    
    let extracted = []; 

    for (const comment of required_comments) {
        const content = await comment.$$eval('*', elements => {
            return elements.map(el => {
                // Check if it's an image inside a span
                if (el.tagName === 'IMG' && el.alt) {
                    return { type: 'image', content: el.alt.trim() };
                } 
                // If it's a span, extract its inner text
                else if (el.tagName === 'SPAN' && el.innerText.trim()) {
                    return { type: 'span', content: el.innerText.trim() };
                } 
                // If it's a regular element with text, extract the text
                else if (el.innerText.trim()) {
                    return { type: 'text', content: el.innerText.trim() };
                }
            }).filter(item => item); // Filter out undefined or empty items
        });

        extracted.push(content);
    }

    console.log(extracted);
    return extracted;
}


async function getContentSolution(divElement) {
    const text = await divElement.evaluate(div => div.textContent);

    console.log('Extracted text:', text);
}

// Usage example
// const test_url = "https://artofproblemsolving.com/community/c3922196_2023_isl";
// const test_class_name = "cmty-view-post-topic-link";
// scrapeDivs(test_url, test_class_name);
// Testing comment scraping
const comment_class = "cmty-post-html"
const link = "https://artofproblemsolving.com/community/c6h3106752p28097575"

async function test() {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    await page.goto(link);
    await getNComments(page, comment_class, n=1);
    await browser.close();
}
test();
