# CV PDF Instructions

The CV link in the navigation now points directly to `/files/cv.pdf`.

## How to Update Your CV

1. Create your CV as a PDF file (you can use tools like:
   - LaTeX (recommended for academic CVs)
   - Microsoft Word or Google Docs (export as PDF)
   - Online CV builders
   - Any other tool that can export to PDF)

2. Name your CV file `cv.pdf`

3. Replace the template file at `/files/cv.pdf` with your actual CV PDF

## Current Setup

- **Navigation Link**: The "CV" link in the site header now points to `/files/cv.pdf`
- **Template CV**: A template CV has been created based on the original `cv.md` content
- **Original Page**: The original markdown CV page at `_pages/cv.md` is still available if needed

## Reverting to Page Format

If you want to switch back to using a page instead of a PDF:

1. Edit `_data/navigation.yml`
2. Change the CV url from `/files/cv.pdf` back to `/cv/`
3. Remove or rename the `cv.pdf` file in the files directory
