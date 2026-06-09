# Self-contained RelationalRAG-Tables artifact image (NumPy + scikit-learn core).
# The synthetic core is CPU-only and reproduces every controlled-study and
# diagnostics result. Build from this folder:
#   docker build -t relationalrag-artifact .
#   docker run --rm relationalrag-artifact
# The optional real-data layer (relational_rag_real/) needs an Apple-silicon GPU
# (MPS + MLX) and public benchmark downloads, so it is NOT part of this image.
FROM python:3.12-slim
WORKDIR /workspace/relationalrag
COPY . .
RUN python -m pip install --upgrade pip && python -m pip install -e '.[dev]'
RUN python check_artifact.py
CMD ["bash", "run_smoke.sh"]
