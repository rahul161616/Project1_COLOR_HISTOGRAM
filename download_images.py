from icrawler.builtin import BingImageCrawler

crawler1 = BingImageCrawler(storage={'root_dir': 'dataset/class1'})
crawler1.crawl(keyword='beach landscape', max_num=100)

crawler2 = BingImageCrawler(storage={'root_dir': 'dataset/class2'})
crawler2.crawl(keyword='forest landscape', max_num=100)