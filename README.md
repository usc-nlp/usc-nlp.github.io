# USC NLP Website

Source for [nlp.usc.edu](https://nlp.usc.edu). Plain HTML + CSS, no build step. Edit `index.html` directly and push to `gh-pages`.

## File structure

```
index.html          Single-page site (about, labs, affiliated, alumni, research, calendar)
style.css           Stylesheet
favicon.ico         Favicon
graph/
  data/             Research graph data (interests.json)
  fetch_interests.py  Script to regenerate graph data from Semantic Scholar
images/
  faculty/          Faculty headshots (firstname.jpg)
  students/         Student headshots (firstname.jpg)
  labs/             Lab logos (labname.png)
  site/             Site assets (header/footer logos)
```

## Sections

The page is a single HTML file with anchor navigation:

- **About** (`#about`) — one-paragraph description of the group
- **Labs** (`#labs`) — five main labs, each with logo, PI, description, and student grid
- **Affiliated** (`#affiliated`) — collapsible section for groups without a CS dept PI
- **Alumni** (`#alumni`) — collapsible list of graduated students and their positions
- **Research** (`#research`) — interactive D3 co-authorship graph with lab filter buttons
- **Calendar** (`#calendar`) — embedded Google Calendar

## How to add a PhD student

1. **Add their photo** to `images/students/`. Use a square crop, name it `firstname.jpg` (lowercase, no spaces).

2. **Add their card** to the correct lab's `<div class="students-grid">` in `index.html`:

```html
<a href="https://their-website.com/" class="student-card">
  <img src="images/students/firstname.jpg" alt="First Last">
  <span class="name">First Last</span>
</a>
```

3. **Co-advised students** should be duplicated in every lab they belong to.

4. **Keep alphabetical order by last name** within each lab's student grid.

## How to move a student to alumni

1. **Remove** their `<a class="student-card">...</a>` block from all labs they appear in.

2. **Add a line** to the alumni list in the `<ul class="alumni-list">` section:

```html
<li><a href="https://their-website.com/">First Last</a> <span class="arrow">&rarr;</span> Title @ Company</li>
```

3. **Keep alphabetical order by last name** in the alumni list.

4. You can leave their photo in `images/students/` (it won't be displayed).

## How to add a new lab

Add a new `<div class="lab-group">` block before `</section>` in the Labs section:

```html
<!-- ========== LAB NAME ========== -->
<div class="lab-group">
  <div class="lab-group-header">
    <a href="https://lab-website.com/"><img src="images/labs/labname.png" alt="Lab Name" class="lab-logo"></a>
    <div class="lab-group-info">
      <div class="lab-group-name"><a href="https://lab-website.com/">Lab Name</a></div>
      <p class="lab-desc">One-sentence description of the lab's research.</p>
    </div>
    <div class="faculty-card">
      <a href="https://faculty-website.com/"><img src="images/faculty/firstname.jpg" alt="Faculty Name"></a>
      <div>
        <h4><a href="https://faculty-website.com/">Faculty Name</a></h4>
        <p class="title">Assistant Professor</p>
      </div>
    </div>
  </div>
  <div class="students-grid">
    <!-- student cards go here -->
  </div>
</div>
<!-- ========== END LAB NAME ========== -->
```

## Research graph

The interactive D3 graph in the Research section is powered by `graph/data/interests.json`. To regenerate it:

```sh
python graph/fetch_interests.py
```

This fetches co-authorship and topic data from Semantic Scholar and clusters members by research similarity.

## Style guide

- **Alphabetical order**: Students within each lab and alumni entries are sorted by last name.
- **Photos**: Square crop, reasonable file size. Name files `firstname.jpg` (lowercase).
- **Links**: Every person should link to their personal website. Use `#` if they don't have one.
- **Co-advised students**: Duplicate the student card in each lab. Keep both copies in sync.
- **Affiliated groups**: Labs without a CS department faculty PI go in the collapsible "Affiliated Groups" section.
- **No build step**: Edit HTML directly. Open `index.html` in a browser to preview.
