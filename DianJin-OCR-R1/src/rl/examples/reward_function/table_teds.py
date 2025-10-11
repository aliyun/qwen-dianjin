import re
from typing import Dict, List
import Levenshtein
from apted import APTED, Config
from apted.helpers import Tree
from lxml import etree
import lxml.html as lxml_html
import html
from collections import deque
import unicodedata
from bs4 import BeautifulSoup



class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        """Show tree using brackets notation"""
        if self.tag == 'td':
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % \
                     (self.tag, self.colspan, self.rowspan, self.content)
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


class CustomConfig(Config):
    @staticmethod
    def maximum(*sequences):
        """Get maximum possible value
        """
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        """Get distance from 0 to 1
        """
        return float(Levenshtein.distance(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2):
        """Compares attributes of trees"""
        if (node1.tag != node2.tag) or (node1.colspan != node2.colspan) or (node1.rowspan != node2.rowspan):
            return 1.
        if node1.tag == 'td':
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.


class TEDS(object):
    ''' Tree Edit Distance basead Similarity
    '''
    def __init__(self, structure_only=False, n_jobs=1, ignore_nodes=None):
        assert isinstance(n_jobs, int) and (n_jobs >= 1), 'n_jobs must be an integer greather than 1'
        self.structure_only = structure_only
        self.n_jobs = n_jobs
        self.ignore_nodes = ignore_nodes
        self.__tokens__ = []

    def tokenize(self, node):
        ''' Tokenizes table cells
        '''
        self.__tokens__.append('<%s>' % node.tag)
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != 'unk':
            self.__tokens__.append('</%s>' % node.tag)
        if node.tag != 'td' and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        ''' Converts HTML tree to the format required by apted
        '''
        global __tokens__
        if node.tag == 'td':
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(node.tag,
                                 int(node.attrib.get('colspan', '1')),
                                 int(node.attrib.get('rowspan', '1')),
                                 cell, *deque())
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != 'td':
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node

    def evaluate(self, pred, true, anchor_point='//table'):
        ''' Computes TEDS score between the prediction and the ground truth of a
            given sample
        '''
        if (not pred) or (not true):
            return 0.0
        parser = lxml_html.HTMLParser(remove_comments=True, encoding='utf-8')
        pred = lxml_html.fromstring(pred, parser=parser)
        true = lxml_html.fromstring(true, parser=parser)
        if pred.xpath(anchor_point) and true.xpath(anchor_point):
            pred = pred.xpath(anchor_point)[0]
            true = true.xpath(anchor_point)[0]
            if self.ignore_nodes:
                etree.strip_tags(pred, *self.ignore_nodes)
                etree.strip_tags(true, *self.ignore_nodes)
            n_nodes_pred = len(pred.xpath(".//*"))
            n_nodes_true = len(true.xpath(".//*"))
            n_nodes = max(n_nodes_pred, n_nodes_true)
            try:
                tree_pred = self.load_html_tree(pred)
                tree_true = self.load_html_tree(true)
                distance = APTED(tree_pred, tree_true, CustomConfig()).compute_edit_distance()
                return 1.0 - (float(distance) / n_nodes)
            except:
                return 0.0
        else:
            return 0.0

 
def normalized_html_table(text):
    def process_table_html(md_i):
        """
        pred_md format edit
        """
        def process_table_html(html_content):
            soup = BeautifulSoup(html_content, 'html.parser')
            th_tags = soup.find_all('th')
            for th in th_tags:
                th.name = 'td'
            thead_tags = soup.find_all('thead')
            for thead in thead_tags:
                thead.unwrap()  # unwrap()会移除标签但保留其内容
            math_tags = soup.find_all('math')
            for math_tag in math_tags:
                alttext = math_tag.get('alttext', '')
                alttext = f'${alttext}$'
                if alttext:
                    math_tag.replace_with(alttext)
            span_tags = soup.find_all('span')
            for span in span_tags:
                span.unwrap()
            return str(soup)

        table_res=''
        table_res_no_space=''
        if isinstance(md_i,list): md_i = md_i[0] # TODO ******* ToCheck
        if '<table' in md_i.replace(" ","").replace("'",'"'):
            md_i = process_table_html(md_i)
            table_res = html.unescape(md_i).replace('\n', '')
            table_res = unicodedata.normalize('NFKC', table_res).strip()
            pattern = r'<table\b[^>]*>(.*)</table>'
            tables = re.findall(pattern, table_res, re.DOTALL | re.IGNORECASE)
            table_res = ''.join(tables)
            # table_res = re.sub('<table.*?>','',table_res)
            table_res = re.sub('( style=".*?")', "", table_res)
            table_res = re.sub('( height=".*?")', "", table_res)
            table_res = re.sub('( width=".*?")', "", table_res)
            table_res = re.sub('( align=".*?")', "", table_res)
            table_res = re.sub('( class=".*?")', "", table_res)
            table_res = re.sub('</?tbody>',"",table_res)
            
            table_res = re.sub(r'\s+', " ", table_res)
            table_res_no_space = '<html><body><table border="1" >' + table_res.replace(' ','') + '</table></body></html>'
            # table_res_no_space = re.sub(' (style=".*?")',"",table_res_no_space)
            # table_res_no_space = re.sub(r'[ ]', " ", table_res_no_space)
            table_res_no_space = re.sub('colspan="', ' colspan="', table_res_no_space)
            table_res_no_space = re.sub('rowspan="', ' rowspan="', table_res_no_space)
            table_res_no_space = re.sub('border="', ' border="', table_res_no_space)

            table_res = '<html><body><table border="1" >' + table_res + '</table></body></html>'
            # table_flow.append(table_res)
            # table_flow_no_space.append(table_res_no_space)

        return table_res, table_res_no_space
    
    def clean_table(input_str,flag=True):
        if flag:
            input_str = input_str.replace('<sup>', '').replace('</sup>', '')
            input_str = input_str.replace('<sub>', '').replace('</sub>', '')
            input_str = input_str.replace('<span>', '').replace('</span>', '')
            input_str = input_str.replace('<div>', '').replace('</div>', '')
            input_str = input_str.replace('<p>', '').replace('</p>', '')
            input_str = input_str.replace('<spandata-span-identity="">', '')
            input_str = re.sub('<colgroup>.*?</colgroup>','',input_str)
        return input_str
    try:
        norm_text, _ = process_table_html(text)
        norm_text = clean_table(norm_text) 
    except Exception as e:
        # with open('./error.log','a+') as f:
        #     f.write(str(e)+'\n')
        norm_text = text
    return norm_text

 
def answer_match_teds(predict, ground_truth):
    teds = TEDS(n_jobs=4)
    score = teds.evaluate(normalized_html_table(predict), normalized_html_table(ground_truth) )
    return score


def extract_table(text):
    pattern = r'<table.*?</table>'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        last_table = matches[-1]  # 取最后一个匹配
        return last_table
    else:
        return ''

def format_reward(predict: str) -> float:
    # pattern = re.compile(r"<think>.*?</think>.*?```html.*?```", re.DOTALL)
    pattern = re.compile(r"```html.*?```", re.DOTALL)
    format_match = re.fullmatch(pattern, predict)
    return 1.0 if format_match else 0.0

def format_reward_withThink(predict: str) -> float:
    pattern = re.compile(r"<think>.*?</think>.*?```html.*?```", re.DOTALL)
    format_match = re.fullmatch(pattern, predict)
    return 1.0 if format_match else 0.0




def compute_score_table(reward_inputs, format_weight: float = 0.1) -> List[Dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        ground_truth = reward_input["ground_truth"]
        
        format_score = format_reward(predict)
        pred_html = extract_table(predict)
        accuracy_score = answer_match_teds(pred_html, ground_truth)
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )
    return scores


# * --------------- *
def compute_score_table_withThink(reward_inputs, format_weight: float = 0.1) -> List[Dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        ground_truth = reward_input["ground_truth"]
        
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)  # handle qwen2.5vl-32b format
        format_score = format_reward_withThink(predict)
        pred_html = extract_table(predict)
        accuracy_score = answer_match_teds(pred_html, ground_truth)
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )
    return scores



# * ----------------   THINK FORMAT  ------------- *
def ocr_analysis_format_reward(response):
    #  tool (1') / format(table | rethink ｜answer) (1')
    try:
        # Extract sections from response
        recognition = re.search(r'<recognition>(.*?)</recognition>', response, re.DOTALL)
        think = re.search(r'<rethink>(.*?)</rethink>', response, re.DOTALL)
        final_answer = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        
        
        if not (recognition and think and final_answer):
            format_score = 1.0
            recognition_text = recognition.group(1).strip()
            answer_text = final_answer.group(1).strip()
            # Split answer into components
            # Rule 1: <recognition>    </recognition>  table format
            if len( extract_table(recognition_text) ) == 0:
                format_score = 0.0
            if len(extract_table(answer_text))== 0:
                format_score = 0.0
        else:
            format_score = 0.0
            
        function_call = re.search(r'<function>(.*?)</function>', response, re.DOTALL)
        if function_call:
            func_score = 1.0
        else:
            func_score = 0.0
        return (format_score + func_score) / 2    
    except:
        return 0.0


def compute_score_table_withFormatThink(reward_inputs, format_weight: float = 0.15) -> List[Dict[str, float]]:
    # recognition (包含 <table> </table>) , rethink,  tool_call (调用正确，且有)
    scores = []
    for reward_input in reward_inputs:
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        ground_truth = reward_input["ground_truth"]
        
        format_score = ocr_analysis_format_reward(predict)
        pred_html = extract_table(predict)
        try:
            accuracy_score = answer_match_teds(pred_html, ground_truth)
        except:
            accuracy_score = 0.0
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )
    return scores





def format_reward(response):
    must_shown_words = [
        "<think>",
        "</think>",
        "<rethink>",
        "</rethink>",
        "<answer>",
        "</answer>",
    ]
    for word in must_shown_words:
        if word not in response:
            return 0.0
    # Extract sections from response
    return 1.0


def compute_score_table_withEasyFormat(reward_inputs, format_weight: float = 0.15) -> List[Dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        ground_truth = reward_input["ground_truth"]
        
        format_score = format_reward(predict)
        pred_html = extract_table(predict)
        try:
            accuracy_score = answer_match_teds(pred_html, ground_truth)
        except:
            accuracy_score = 0.0
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )
    return scores
