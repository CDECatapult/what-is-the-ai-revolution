Analysis of the AI publishing landscape. See [this notebook](https://github.com/CDECatapult/what-is-the-ai-revolution/blob/master/Why%20the%20'AI%20revolution'%20is%20really%20a%20deep%20learning%20revolution.ipynb) in the repository

To run the notebook and script:

Create a new anaconda environment:

```python
conda create -n revolution anaconda
```

activate it

```python
source activate revolution
```

Download arXiv data into data.jsonlines (takes about 1 hour)

```python
python scrape.py > data.jsonlines
```

Download Guardian data into guardian.jsonlines (you will need to insert your API key into `creds_guardian.txt`)

```python
python guardian.py > data.jsonlines
```

To run the notebooks

```python
jupyter notebook
```
