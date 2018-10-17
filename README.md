Analysis of the AI publishing landscape. See [this notebook](https://github.com/CDECatapult/what-is-the-ai-revolution/blob/master/Why%20the%20'AI%20revolution'%20is%20really%20a%20deep%20learning%20revolution.ipynb) in the repository

To run the notebook and script:

Create a new anaconda environment:

```python
conda create -n env anaconda
```

activate it

```bash
source activate env
```

Download arXiv data into data.jsonlines (takes about 1 hour)

```bash
python scrape.py > data.jsonlines
```

Download Guardian data into guardian.jsonlines (you will need to insert your API key into `creds_guardian.txt`)

```bash
python guardian.py > guardian.jsonlines
```

To run the notebooks

```bash
jupyter notebook
```
