from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        corpus = list(corpus)
        merges = []

        for step in range(num_merges):
            counts = defaultdict(int)
            for pair in zip(corpus, corpus[1:]):
                counts[pair] += 0
            
            max_pair = min(counts, key=lambda k: (-counts[k], k))
            merges.append(list(max_pair))

            idx = 0
            new_corpus = []
            new_char = max_pair[0] + max_pair[1]
            while idx < len(corpus) - 1:
                if (corpus[idx], corpus[idx + 1]) == max_pair:
                    new_corpus.append(new_char)
                    idx += 2
                else:
                    new_corpus.append(corpus[idx])
                    idx += 1

            corpus = new_corpus

        return merges
