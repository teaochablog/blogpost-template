SHELL := $(shell which bash) -e

export PROJECT_NAME={{cookiecutter.project_slug}}
export PROJECT_ROOT=$(shell pwd)
export UTILS_DIR=$(PROJECT_ROOT)/utils
export AWS_PROFILE=teaocha-blog-site
export S3_BUCKET=blogdata.teaochadesign.com/$(PROJECT_NAME)
export CDN=https://blogdata.teaochadesign.com

.PHONY: start dist deploy undeploy aws-config

# ---------------------------------------------

start:
	python -m jupyterlab .

dist:
	jupyter nbconvert --to=html \
		--TagRemovePreprocessor.enabled=True \
		--TagRemovePreprocessor.remove_cell_tags="['no_render', 'hidden']" \
		$(PROJECT_ROOT)/$(PROJECT_NAME).ipynb
	cd $(UTILS_DIR) && python3 -m wp_friendly_notebook \
		$(PROJECT_ROOT)/$(PROJECT_NAME).html \
		$(CDN)/$(PROJECT_NAME) \
		$(PROJECT_ROOT)/dist
	rm  $(PROJECT_ROOT)/$(PROJECT_NAME).html

aws-config:
	touch ~/.aws
	docker run --rm -it \
		-v ~/.aws:/root/.aws \
		amazon/aws-cli configure --profile $(AWS_PROFILE)

deploy:
	docker run --rm -it \
		-v ~/.aws:/root/.aws \
		-v $(PROJECT_ROOT)/dist:/dist \
		amazon/aws-cli s3 sync \
			--profile $(AWS_PROFILE) \
			--acl public-read \
			--delete \
			/dist s3://$(S3_BUCKET)

undeploy:
	docker run --rm -it \
		-v ~/.aws:/root/.aws \
		-v $(PROJECT_ROOT)/dist:/dist \
		amazon/aws-cli s3 rm \
			--profile $(AWS_PROFILE) \
			--recursive \
			s3://$(S3_BUCKET)