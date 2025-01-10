from bs4 import BeautifulSoup
import bs4
from torch.utils.data import Dataset
import pickle
import os
from tqdm import tqdm
from urllib.parse import urlparse
from transformers import LongformerTokenizerFast
from tld import get_fld
from collections import defaultdict
import traceback

class ContentDataset(Dataset):
    def __init__(self, filepaths, url_list=None, cache_path='../cache/combine_dataloader_no_brand.pkl', force_build=False, inference_mode=False):
        '''
        if inference_mode == False:
            get every HTML file in filepaths, create data, save it in cache_path
        else:
            filepaths -> list of domains
        '''
    
        self.inference_mode = inference_mode
        self.dataset = []
        self.labels = []
        self.url = []
        self.data_html = []
        self.data_body = []
        visited = set()
        visited_counter = 0

        tag_len = []
        
        self.tags = ['a', 'abbr', 'acronym', 'address', 'applet', 'area', 'article', 'aside', 'audio', 'b', 'base', 'basefont', 'bdi','bdo', 'bgsound', 'big', 'blink', 'blockquote', 'body', 'br', 'button', 'canvas', 'caption', 'center', 'circle', 'cite', 'clipPath','code', 'col', 'colgroup', 'command', 'content', 'data', 'datalist', 'dd', 'defs', 'del', 'details', 'dfn', 'dialog', 'dir', 'div','dl', 'dt', 'element', 'ellipse', 'em', 'embed', 'fieldset', 'figcaption', 'figure', 'font', 'footer', 'foreignObject', 'form','frame', 'frameset', 'g', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'head', 'header', 'hgroup', 'hr','html','i','iframe','image','img','input','ins','isindex','kbd','keygen','label','legend','li','line','linearGradient','link','listing','main','map','mark','marquee','mask','math','menu','menuitem','meta','meter','multicol','nav','nextid','nobr','noembed','noframes','noscript','object','ol','optgroup','option','output','p','param','path','pattern','picture','plaintext','polygon','polyline','pre','progress','q','radialGradient','rb','rbc','rect','rp','rt','rtc','ruby', 's', 'samp', 'script', 'section', 'select', 'shadow', 'slot', 'small', 'source', 'spacer', 'span', 'stop', 'strike', 'strong', 'style', 'sub', 'summary', 'sup', 'svg', 'table', 'tbody', 'td', 'template', 'text', 'textarea', 'tfoot', 'th', 'thead', 'time', 'title', 'tr', 'track', 'tspan', 'tt', 'u', 'ul', 'var', 'video', 'wbr', 'xmp']
        self.tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
        url_list_no_slash = set([i.replace('/', '') for i in url_list])
        url_list = set(url_list)
        if not inference_mode and os.path.exists(cache_path) and force_build == False :
            self.data_html, self.data_body, self.url, self.labels = pickle.load( open(cache_path, 'rb'))
        else:
            for directory, label in filepaths.items():
                all_fnames = set(os.listdir(directory))
                for url in tqdm(list(url_list)):
                    if url + '.html' in all_fnames:
                        filename = url + '.html'
                    else:
                        continue
                    filepath = os.path.join(directory, filename)
                    # check if fname is in urls_list
                    if filename.replace('.html', '') in url_list or filename.replace('.html', '') in url_list_no_slash:
                        with open(filepath, 'r', encoding='utf-8') as fin:
                            try:  
                                # read content of file from .html
                                data = str(fin.read())
                                URL = filename.replace('.html', '').replace('https','')
                                domain = self.extract_domain(URL)

                                if domain in visited:
                                    visited_counter += 1

                                content = self.extract_txt(data)

                                if domain not in visited:
                                    if len(content) < 200:
                                        continue

                        
                                    html_traverse = self.walker(BeautifulSoup(data))

                                    tag_ids = [self.tag_to_ind(tag) for tag in html_traverse.split(' ')]
                                    tag_len.append(len(tag_ids))
                                    tag_ids= tag_ids + [len(self.tags) + 1]*(1024-len(tag_ids)) if len(tag_ids) < 1024 else tag_ids[:1024]

                                    # inputs = self.tokenizer.encode_plus(content.lower(), padding="max_length", truncation= True, max_length=1024)

                                    html, body = tag_ids, content.lower()

                                    self.data_html.append(html)
                                    self.data_body.append(body)
                                    self.url.append(URL)
                                    self.labels.append(label)

                                    visited.add(domain)
                            except Exception as e:
                                print(URL, e)
                                traceback.print_exception(type(e), e, e.__traceback__)
                                continue
            
            self.visited = visited
            print('Domain Visited = %d | %d' %(len(self.visited), visited_counter))
            print('Data Len = %d' %len(self.url))
            if not inference_mode:
                with open(cache_path, 'wb') as handle:
                    pickle.dump((self.data_html, self.data_body, self.url, self.labels), handle)

        data_inds = defaultdict(list)  
        for ind, label in enumerate(self.labels):
            data_inds[label].append(ind)

        print('legit count:\t%d  |  fake count:\t%d' %(len(data_inds[0]), len(data_inds[1])))
  
       
    def __len__(self):
        return len(self.data_body)

    def __getitem__(self, idx):
        # return torch.tensor(self.data_html[idx]), torch.tensor(self.data_body[idx]), torch.tensor(self.labels[idx]), self.url[idx]
        if not self.inference_mode:
            return self.data_body[idx], self.labels[idx]
        return self.data_body[idx], self.labels[idx], self.url[idx]



    def extract_domain(self, url):
        '''
        if url does not starts with http, urlparse returns the whole url as 'path' and netloc would be empty
        '''
        try:
            return get_fld(self.fix_url(url))
        except: 
            return urlparse(self.fix_url(url)).netloc

    def fix_url(self, url):
        if not (url.startswith('http://') or url.startswith('https://')):
            url = 'http://{}'.format(url)
        return url

    def walker(self, soup):
        s = ''
        if soup.name is not None:
            for child in soup.children:
                #process node
                if type(child) is bs4.element.Tag:
                    s += str(child.name) + ' ' + self.walker(child)
        return s

    def tag_to_ind(self, tag):
        if tag == '' or tag not in self.tags:
            return len(self.tags)
        return self.tags.index(tag)
    
    def extract_txt(self, html_text):
        soup = BeautifulSoup(html_text, 'html.parser')
        # kill all script and style elements
        for script in soup(["script", "style"]):
            script.extract()    # rip it out

        # get text
        text = soup.get_text()

        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return text

