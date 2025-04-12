const puppeteer = require('puppeteer');
const fs = require('fs');  // Import the fs module

async function scrapeAOPSLink(link, file_path = "problems.json", n = 2) {
    const problem_big_div = "cmty-view-posts-item.cmty-view-post-item-w-label.cmty-vp-both";
    const problem_name_div = "cmty-view-post-item-label";
    const problem_topic_div = "cmty-view-post-topic-link";
    const comment_class = "cmty-post-html";

    // Launch the browser and open a new page
    const browser = await puppeteer.launch();
    const page = await browser.newPage();

    // Navigate to the URL
    await page.goto(link);
    console.log("Accessed the link...")

    // Wait for the divs with the specified class to load
    await page.waitForSelector(`.${problem_big_div}`);

    // Extract all divs with the specified class name
    let result = [];
    const divElements = (await page.$$(`.${problem_big_div}`));
    console.log("Got the divs...");

    // Iterate over each div element and call the processDiv function
    for (const divElement of divElements) {
        const name = await divElement.$eval(`.${problem_name_div}`, el => el.textContent);
        const link = await divElement.$eval(`.${problem_topic_div} a`, el => el.href);
        console.log(`Name: ${name}\nLink: ${link}`);
        
        const new_page = await browser.newPage();
        await new_page.goto(link);
        discussions = await getNComments(new_page, comment_class, n);

        const scrape_data = {
            "Name": name,
            "Link": link,
            "Discussions": discussions
        };
        result.push(scrape_data);
    }
    fs.writeFileSync(file_path, JSON.stringify(result, null, 2));
    console.log(`Extracted problems saved to ${file_path}`);
    await browser.close();
}

async function getNComments(page, div_class, n = 2) {
    await page.waitForSelector(`.${div_class}`);
    const comments = await page.$$(`.${div_class}`);
    
    // Slice to get only the required number of comments
    const required_comments = comments.slice(0, n);
    
    let extracted = []; 

    for (const comment of required_comments) {
        const commentHTML = await comment.evaluate(el => el.outerHTML);
        extracted.push(commentHTML);
    }
    await page.close();
    return extracted;
}

async function scrapeForLink(links, savePath, n = 10) {
    // Read the JSON file containing links
    const rawData = fs.readFileSync(jsonFilePath);
    const linksData = JSON.parse(rawData);

    // Iterate over each link in the JSON file and scrape data
    for (const linkData of linksData) {
        // Assuming linksData contains an array of objects with 'Link' property
        const link = linkData.Link;  // Adjust the key if it's different
        console.log(`Scraping link: ${link}`);
        await scrapeAOPSLink(link, savePath, n);  // Call the scraping function
    }
}

async function processLinksOneByOne() {
    const links_path = './imo_shortlists_links.json';
    const raw_data = fs.readFileSync(links_path);
    const datas = JSON.parse(raw_data);
    const n = 10;

    for (const data of datas) {
        const name = data['title'];
        const link = data['url'];
        const save_path = `./shortlists_raw_htmls/${name}.json`;
        console.log(`Processing ${name} by link ${link}.`);
        await scrapeAOPSLink(link, save_path, n)
    }
}

processLinksOneByOne()
    .then(() => console.log('All links processed successfully!'))
    .catch(err => console.error('Error processing links:', err));
    