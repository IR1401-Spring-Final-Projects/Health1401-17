let button = document.getElementById("cta-btn");
let option = document.getElementById("select");
let query = document.getElementById("query");
let dataSection = document.getElementById("data-section");

function snippStructure(title, abstract, link, tags) {
    return `
    <div class="doc">
        <h2><a href="${link}">${title}</a></h2>
        <p>${abstract}</p>
        <p><span class="tag">دسته‌بندی ها: </span> ${tags.reduce((prev, cur) => cur + "، " + prev)}</p>
      </div>
    `;
}

function printDocs(data) {
    dataSection.innerHTML = "";
    console.log(typeof data)
    if (data['isClassification']===true){
        dataSection.innerHTML = `<div class="doc">کوئری شما در طبقه <span class="tag"> ${data['result']} </span>قرار میگیرد.</div>`
        return
    }
    if (option.value==="cluster"){
        dataSection.innerHTML = "<p>5 داک از خوشه‌ای که کوئری شما در آن قرار گرفته است مطابق زیر می‌باشد.</p>"
    }
    for (doc of data) {
        let title = doc.title;
        let abstract = doc.abstract || doc.paragraphs;
        let link = doc.link;
        let tags = doc.tags || doc.categories;
        dataSection.innerHTML += snippStructure(title, abstract, link, tags);
    }
}

button.addEventListener("click", () => {
    fetch(`http://localhost:8000/result?query=${query.value}&action_type=${option.value}`)
    .then(res => res.json())
.then(data => {printDocs(data)})
.catch(err => console.log(err));
});