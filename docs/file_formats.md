# File Formats

This document describes the various file formats used in the Academic IR Pipeline.

## Scraped Data

-   `scraped_data.json`: Contains raw scraped content

    ```json
    [
      {
        "url": "https://example.edu/page",
        "title": "Page Title",
        "meta_description": "Page description",
        "content": "Main content..."
      }
    ]
    ```

## Corpus Files

- `passages.csv`: Main corpus file with all passages

This file contains the segmented passages used for indexing and retrieval.

| id                                  | text (excerpt)                                                                                                           | title                                                      | url                                                                 | start_idx | end_idx | #sentences | #words | source            |
|-------------------------------------|--------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|----------------------------------------------------------------------|-----------|----------|-------------|--------|-------------------|
| 0bb1b080188149727831a663a97b077a    | "international.mst.edu Sponsored Student Services...Invoices for tuition and fees will be sent directly..."              | Current Students – International Student and Scholar Services | https://international.mst.edu/sponsored-student-services/...         | 0         | 9        | 10          | 192    | scraped_data.json |
| 4016a0b28cec8997e75facb8a72f97b2    | "Event Calendar Click here to view our Event Calendar...maintain your immigration status. Sponsor Requests..."           | Current Students – International Student and Scholar Services | https://international.mst.edu/sponsored-student-services/...         | 0         | 9        | 10          | 250    | scraped_data.json |

📌 *Only the first two rows are shown. The `text` column is truncated for readability.*

📥 Download Programmatically (Python)
```python
from huggingface_hub import hf_hub_download

file_path = hf_hub_download(
    repo_id="ShubhamC/rag-tutorial-prebuilt-indexes",
    filename="custom_mst_site/passages.csv",
    repo_type="dataset"
)
print(f"File downloaded to: {file_path}")
```

-   `metadata.json`: Document-level metadata

## Topics

We provide a `topics.json` file which contains NIST-style topic definitions.

### Example

     ```json
     {
      "id": "1",
      "title": "What is the deadline for dropping a course without receiving a 'W' grade?",
      "description": "An undergraduate student wants to know the last day they can drop a course without having a withdrawal mark on their transcript.",
      "narrative": "Relevant information should include the specific deadline date for the current or upcoming semester, any differences between undergraduate and graduate policies, and the procedural steps for dropping a course. Information about dropping with a 'W' grade or complete semester withdrawals is not relevant unless it clearly differentiates the \"no W\" deadline."
    },
    {
      "id": "2",
      "title": "How do I apply for a teaching assistantship in the Computer Science department?",
      "description": "A graduate student is interested in becoming a teaching assistant and needs to know the application process.",
      "narrative": "Relevant documents should outline eligibility requirements, application deadlines, required materials (CV, statement of interest, etc.), selection criteria, expected time commitments, and compensation details. Information about research assistantships or TA positions in other departments is not relevant."
    },
    {
      "id": "3",
      "title": "What mental health resources are available to undergraduate students through the university?",
      "description": "An undergraduate student is experiencing stress and anxiety and wants to know what support services the university offers.",
      "narrative": "Relevant information should detail counseling services, crisis intervention options, support groups, wellness programs, and how to access these services (appointment procedures, walk-in hours, online resources) for undergraduate students. Information should include contact information, location, hours of operation, and whether services are free or fee-based. General health services without mental health components are not relevant."
    },
    {
      "id": "4",
      "title": "How can I reserve study rooms in the Science Library?",
      "description": "A group of students needs to reserve a study space for collaborative project work.",
      "narrative": "Relevant documents should explain the room reservation system, including how to check availability, make reservations online or in-person, maximum reservation time, group size limitations, available technology/equipment in rooms, and any policies regarding cancellations or no-shows. Information about other libraries or general library hours without reservation details is not relevant."
    }
     ```

## Relevance Judgements

We provide `qrels.txt` which contains TREC-style relevance judgments indicating how relevant each passage is to a given topic.

### Format (per line)
```
<topic_id> 0 <passage_id> <relevance_score>
```
where:
- `topic_id`: ID of the topic or question
- `passage_id`: ID of the passage from `passages.csv`
- `relevance_score`:
   - 0 = not relevant
   - 1 = somewhat relevant
   - 2 = highly relevant
     
### Example

```
1 0 0bb1b080188149727831a663a97b077a 1
1 0 4016a0b28cec8997e75facb8a72f97b2 2
1 0 494c6d2386b8b76ba262ea524d44c56f 2
1 0 ee52abe83f1e30fe17fa7a26eec97fff 2
 ```
**Explanation**

- The first line says that for topic 1, passage `0bb1b080188149727831a663a97b077a` is somewhat relevant (score = 1).
- The second line says that for the same topic, passage `4016a0b28cec8997e75facb8a72f97b2` is highly relevant (score = 2).
