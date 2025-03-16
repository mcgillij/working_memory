# Working Memory

## Introduction

Use a list of bookmarks, in this case from firefox, to seed a vector database with some context for LLM queries.

## Process

1. Export bookmarks from firefox
2. Parse bookmarks
3. Create SQLite database of links to fetch
4. Fetch links
5. Parse links
6. Mark links as fetched or dead / error
7. Clean up links / extract text only
8. Create a vector database


## some kind of pluggable cleaning modules / or processing / data massaging

examples:

a github connector that would be able to see if the link is a github link and then fetch the readme.md file and add it to the text / possibly some of the source code

a youtube connector that would be able to see if the link is a youtube link and then fetch the transcript and add it to the text

a wikipedia connector that would be able to see if the link is a wikipedia link and then fetch the article and add it to the text

pdf connector that would be able to see if the link is a pdf link and then fetch the text from the pdf and add it to the text
