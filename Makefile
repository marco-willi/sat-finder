.PHONY: help download-vienna download-vienna-stitch download-vienna-full download-vienna-full-hires \
        download-graz download-graz-stitch precompute-embeddings compute-similarity-matrix \
        app app-dev tiles

#################################################################################
# COMMANDS                                                                      #
#################################################################################

help:	## Show this help.
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

##### DATA

download-vienna: ## Download Vienna Orthofoto tiles (central area at zoom 18)
	python scripts/download_vienna_data.py --preset central --zoom 18

download-vienna-stitch: ## Download and stitch Vienna tiles into single image
	python scripts/download_vienna_data.py --preset central --zoom 18 --stitch

download-vienna-full: ## Download full Vienna Orthofoto (zoom 17, ~1.2GB)
	python scripts/download_vienna_full.py --zoom 17

download-vienna-full-hires: ## Download full Vienna Orthofoto high-res (zoom 18, ~4.5GB)
	python scripts/download_vienna_full.py --zoom 18

download-graz: ## Download Graz Orthofoto tiles (central area at zoom 18)
	python scripts/download_graz_data.py --preset central --zoom 18

download-graz-stitch: ## Download and stitch Graz tiles into single image
	python scripts/download_graz_data.py --preset central --zoom 18 --stitch

precompute-embeddings: ## Pre-compute DINOv3 embeddings for satellite images
	python scripts/precompute_embeddings.py

compute-similarity-matrix: ## Compute self-similarity matrix from pre-computed embeddings
	python scripts/compute_similarity_matrix.py \
		--embeddings assets/embeddings.npz \
		--output assets/similarity_matrix.npy \
		--sim-batch-size 64 \
		--device cuda

##### GRADIO APP

app: ## Start Gradio app (production mode with uvicorn)
	python app.py

app-dev: ## Start Gradio app (development mode with auto-reload)
	gradio app.py

tiles: ## Generate DeepZoom tiles from map.jpg
	@echo "Generating DeepZoom tiles..."
	@mkdir -p static/tiles
	vips dzsave assets/map.jpg static/tiles/scene \
		--tile-size 256 \
		--overlap 0 \
		--suffix .jpg[Q=90]
	@echo "Tiles generated at static/tiles/"
