import os
import json
import urllib.request
import multiprocessing
from PIL import Image


# async worker
class Download(object):
    def __init__(self, jobs):
        self.jobs = jobs

    def download(self, idx):
        job = self.jobs[idx]
        url, filename = job
        exists = False
        try:
            image = Image.open(filename)
            if all(image.size):
                exists = True
        except:
            exists = False
        if not exists:
            urllib.request.urlretrieve(url, filename)
        print(f"\t download: {filename}")

    def run(self, num_workers):
        print(f"Multiple threads: {num_workers}")
        with multiprocessing.Pool(processes=num_workers) as pool:
                cnt = 0
                for index in pool.imap_unordered(self.download, range(len(self.jobs))):
                    cnt += 1
                    print(f"Finish: {cnt}/{len(self.jobs)}")


if __name__ == '__main__':
    # Algorithm decompositions to download (indexed by slug).  In our publication
    # [Bell et al 2014], we evaluated these algorithms:
    ALGORITHMS_TO_DOWNLOAD = set((
        'bell2014_densecrf',
        'zhao2012_nonlocal',
        'garces2012_clustering',
        'grosse2009_grayscale_retinex',
        'grosse2009_color_retinex',
        'shen2011_optimization',
        # 'baseline_reflectance',
        # 'baseline_shading',
    ))

    # Set to True to also download full-resolution original input images (18G
    # total).  The downsampled images are already included with the judgements
    # dataset release, so there's no need unless you need full-resolution inputs.
    DOWNLOAD_ORIGINAL_IMAGES = False

    # Prepare list of files to download, by parsing the included JSON file
    algorithms = json.load(open('./intrinsic-decompositions-export.json'))
    jobs = []
    for algorithm in algorithms:
        if algorithm['slug'] in ALGORITHMS_TO_DOWNLOAD:
            print('Downloading: %s' % algorithm['slug'])

            for decomposition in algorithm['intrinsic_images_decompositions']:
                if DOWNLOAD_ORIGINAL_IMAGES:
                    jobs.append((
                        decomposition['original_image'],
                        'original_image/%s.jpg' % (decomposition['photo_id'])))
                jobs.append((
                    decomposition['reflectance_image'],
                    '%s/%s-r.png' % (algorithm['slug'], decomposition['photo_id'])))
                jobs.append((
                    decomposition['shading_image'],
                    '%s/%s-s.png' % (algorithm['slug'], decomposition['photo_id'])))
        else:
            print('NOT downloading: %s' % algorithm['slug'])


    # remove dupliates (this will happen based on the way the loop is set up)
    jobs = list(set(jobs))

    # create directories
    if DOWNLOAD_ORIGINAL_IMAGES and not os.path.isdir('original_image'):
        os.makedirs('original_image')
    for slug in ALGORITHMS_TO_DOWNLOAD:
        if not os.path.isdir(slug):
            os.makedirs(slug)


    # set up pool
    cpu_count = 4  #multiprocessing.cpu_count()
    print('\nDownloading %s images using %s worker processes...' % (len(jobs), cpu_count))
    print("No progress bar, sorry.  Note that existing images are skipped.")

    # Use map_async with a long timeout instead of map -- this avoids a
    # KeyboardInterrupt-related bug in Python.
    # for j in jobs:
    #     download(j)
    # pool = multiprocessing.Pool(cpu_count)
    # pool.map_async(download, jobs).get(999999)
    d = Download(jobs)
    d.run(cpu_count)
    print('Done!')
