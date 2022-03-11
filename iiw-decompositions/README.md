# Intrinsic Images in the Wild - Decompositions

We are releasing precomputed intrinsic image decompositions for all of the
algorithms that we evaluated, as image URLs.  To keep our costs low, please
avoid re-downloading the images too many times :).


## Download script

This script will download all decompositions for our algorithm and save it to
the local directory (`bell2014_densecrf/`):

    ./download_images.py

*Other algorithms*:
By default, the script only downloads decompositions for our algorithm.  Edit
the top of the script by uncommenting the algorithms you also want to download
in the variable `ALGORITHMS_TO_DOWNLOAD`.  The images are stored as sRGB (not
linear) and saved to the filename `(algorithm)/(photo_id)-r.png` for
reflectance and `...-s` for shading.  The decompositions use about 2.1GB of disk
space (per algorithm).

*Full-resolution inputs*:
To also download the input images at full resolution, set
`DOWNLOAD_ORIGINAL_IMAGES = True` in the script.  They will be saved to
`original_image/`.  The downsampled images are already included with the
judgements dataset release, so there's no need unless you need full-resolution
inputs.  The originals use about 18GB at full-resolution.


## Looking up images

Note that all images are indexed by their OpenSurfaces ID (`photo_id`), so you
can look up all attributes (e.g. material segmentations) for an image by
visiting:

`http://opensurfaces.cs.cornell.edu/photos/<photo_id>/`


## JSON file index

The decompositions are also exported as a JSON file
(`intrinsic-decompositions-export.json`) and has the following format,
annotated with inline comments:

    [
      {
        # Internal database ID
        "id": 1141,

        # Unique human-readable ID for this algorithm
        "slug": "bell2014_densecrf",

		# Citation in HTML format
        "citation_html": "Sean Bell, Kavita Bala, Noah Snavely.  \"Intrinsic Images in the Wild\".  <i>ACM Transactions on Graphics (SIGGRAPH 2014)</i>.  <a href=\"http://intrinsic.cs.cornell.edu\">http://intrinsic.cs.cornell.edu</a>.",

        # Mean training error (WHDR) across all images
        "iiw_mean_error": 0.210448995151284,

		# Mean runtime in seconds for all images (see below note about runtime
		# and why these numbers are unreliable).
        "iiw_mean_runtime": 214.436177993732,

        # Parameters used for the algorithm to obtain this training error
        "parameters": {
          "abs_shading_weight": 500.0,
          "n_iters": 25,
          "chromaticity_weight": 0,
          "kmeans_intensity_scale": 0.5,
          "theta_c": 0.025,
          "abs_shading_gray_point": 0.5,
          "pairwise_weight": 10000.0,
          "theta_l": 0.1,
          "theta_p": 0.1,
          "shading_blur_init_method": "none",
          "shading_blur_sigma": 0.1,
          "shading_target_weight": 20000.0,
          "pairwise_intensity_chromaticity": true,
          "split_clusters": true,
          "abs_reflectance_weight": 0,
          "kmeans_n_clusters": 20,
          "shading_target_norm": "L2"
        },

        # List of all decompositions for this algorithm, sorted by photo_id
        "intrinsic_images_decompositions": [
          {
            # Internal database ID
            "id": 2293128,

            # OpenSurfaces photo ID (corresponds to filename in the dataset release)
            "photo_id": 102443,

			# Runtime of this photo in seconds.  Note that these decompositions
			# were run on different machine types with varying load, so this
			# number is not reliable.  Nonetheless, you can probably still compare
			# average orders of magnitude.
            "runtime": 152.869494915009,

            # WHDR for this photo out of 1.0 (across both "sparse" and "dense" judgements)
            "mean_error": 0.0,

            # WHDR for this photo out of 1.0, using only "sparse" judgements, or `null` if there are no sparse edges
            "mean_sparse_error": 0.0,

            # WHDR for this photo out of 1.0, using only "dense" judgements, or `null` if there are no dense edges
            "mean_dense_error": null,

			# Original image at full resolution (as obtained from Flickr).
			# Note that for all of our decompositions, the input images were
			# resized to fit in a 512x512 box before being decomposed.  If
			# downsampling the image, I suggest saving it as PNG since some
			# JPEG artifacts become visible after decomposing.
            "original_image": "http://labelmaterial.s3.amazonaws.com/photos/salvadonica_5104456591.jpg",

            # Reflectance image, sRGB tone-mapped for display (not linear)
            "reflectance_image": "http://labelmaterial.s3.amazonaws.com/intrinsic_reflectance/AcTZeNXobUhG3lMXDf7erYi.png",

            # Shading image, sRGB tone-mapped for display (not linear)
            "shading_image": "http://labelmaterial.s3.amazonaws.com/intrinsic_shading/AcUH22tsJQNOD3eeuKjxkzM.png"

			# Attribution information: Name, URL, License.
			# Note that a few of the photos with varying lighting conditions are not Creative Commons.
            "attribution_name": "Sally",
            "attribution_url": "http://www.flickr.com/photos/quiltsalad/3711222369/",
            "license_name": "Attribution 2.0 Generic",
            "license_url": "http://creativecommons.org/licenses/by/2.0/"
          },

          # remaining photographs
          ...
        ]
      },

      # remaining algorithms
      ...
    ]


## Citation

If you use the decompositions from our algorithm (`bell2014_densecrf`), please
cite our paper:

    Sean Bell, Kavita Bala, Noah Snavely
    Intrinsic Images in the Wild
    ACM Transactions on Graphics (SIGGRAPH 2014)

    @article{bell14intrinsic,
		author = "Sean Bell and Kavita Bala and Noah Snavely",
		title = "Intrinsic Images in the Wild",
		journal = "ACM Trans. on Graphics (SIGGRAPH)",
		volume = "33",
		number = "4",
		year = "2014",
	}


## Questions

Please don't hesitate to ask me any questions or tell me about how you're using
our data: sbell@cs.cornell.edu (Sean Bell)
