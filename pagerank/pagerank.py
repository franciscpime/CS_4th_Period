import os
import random
import re
import sys

DAMPING = 0.85

# number of samples I'll use to estimate PageRank using the sampling method
SAMPLES = 10000


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")

    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)

    print(f"PageRank Results from Sampling (n = {SAMPLES})")

    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")

    ranks = iterate_pagerank(corpus, DAMPING)

    print(f"PageRank Results from Iteration")

    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages

"""
Return a probability distribution over which page to visit next,
given a current page.

With probability `damping_factor`, choose a link at random
linked to by `page`. With probability `1 - damping_factor`, choose
a link at random chosen from all pages in the corpus.
"""
def transition_model(corpus, page, damping_factor) -> dict:
    
    all_pages = {}

    probability = (1 - damping_factor) / len(corpus)

    for p in corpus:
        all_pages[p] = probability

    if corpus[page]:
        extra = damping_factor / len(corpus[page])
        
        for link in corpus[page]:
            all_pages[link] += extra

    else:
        for p in corpus[page]:
            all_pages[p] += probability


    return all_pages


"""
Return PageRank values for each page by sampling `n` pages
according to transition model, starting with a page at random.

Return a dictionary where keys are page names, and values are
their estimated PageRank value (a value between 0 and 1). All
PageRank values should sum to 1.
"""
def sample_pagerank(corpus, damping_factor, n) -> dict:
    n_pages = {p: 0 for p in corpus}

    page = random.choice(list(corpus.keys()))

    for i in range(n):
        n_pages[page] +=1
        probs = transition_model(corpus, page, damping_factor)
        next_page = random.choices(
            list(probs.keys()), 
            weights = probs.values(), 
            k = 1
        )[0]

        page = next_page
    
    return {p: n_pages[p] / n for p in n_pages}


"""
Return PageRank values for each page by iteratively updating
PageRank values until convergence.

Return a dictionary where keys are page names, and values are
their estimated PageRank value (a value between 0 and 1). All
PageRank values should sum to 1.
"""
def iterate_pagerank(corpus, damping_factor):
    
    page_rank = {p: 1 / len(corpus) for p in corpus}

    while True:
        new_page_rank = {}
        for page in corpus:
            soma = 0
            for i in corpus:
                if len(corpus[i]) == 0:  
                    soma += page_rank[i] / len(corpus)
                elif page in corpus[i]:
                    soma += page_rank[i] / len(corpus[i])
            new_page_rank[page] = (1 - damping_factor) / len(corpus) + damping_factor * soma

        if all(abs(new_page_rank[p] - page_rank[p]) < 0.001 for p in corpus):
            break

        page_rank = new_page_rank

    return page_rank

if __name__ == "__main__":
    main()
