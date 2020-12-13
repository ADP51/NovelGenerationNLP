"""
This program is intended to clean html files extracted from the Gutenberg online library.
Some manual processing, particularly at the beginning and end of the file, will be necessary.
Otherwise, this program will remove page numbers, images, redundant line breaks, and other markup elements.

Instructions on use:
Mode 1: Clean a file
-- This mode will accept htm/html file names and create a cleaned text file in the same directory as the source file.
Mode 2: Clean a folder
-- This mode will accept folder names, read all files in the submitted folder, and create clean versions in a
  separate folder
"""

import html
import os
import re
import time
import unidecode
from pathlib import Path

from cchardet import UniversalDetector

TIME = True
COUNT = True

log_time = 0
log_count = 0


# Replace html codes and unicode symbols with respective ascii symbols
def html_edit(line):
    global log_count

    # To match letter codes:
    r"&[a-zA-Z0-9]+;"

    # To match number codes:
    r"&#[0-9]+;"

    # To match all codes:
    r"&#?[a-zA-Z0-9]+?;"

    # To match non-ascii characters
    r"[^\x00-\x7F]"

    # Remove leading spaces/tabs
    line = line.lstrip(' \t')

    # Replace html escape codes with respective symbols
    if COUNT:
        log_count += len(re.findall(r"&#?[a-zA-Z0-9]+?;", line))
    line = html.unescape(line)

    # Replace non-ascii characters with ascii equivalent strings
    if COUNT:
        log_count += len(re.findall(r"[^\x00-\x7F]", line))
    line = unidecode.unidecode(line)

    return line


# delete/replace markup syntax using regex
def regex_edit(text_edit):
    global log_count

    # List of complex expressions to delete matches for - in order of precedence
    list_delete_content = [r"<p><span class=(\"|\')pagenum\1>.*?</span></p>",
                           r"<(a).*?>.*?<\/a>",
                           r"<(img)(.|\n|\r)*?>",
                           r"<span class=(\"|\')(pagenum|pageno|hidden|imgnum)\1.*?>.*?</span>",
                           r"<div class=(\"|\')(fig|figright|figright|figcenter|blockquot|blockquote|title"
                           r"|illus|caption|centered|sidenote)\1.*?>.*?</div>",
                           r"<p class=(\"|\')(ph2|center pfirst|figcenter|figright|figleft|footnote)\1.*?>.*?</p>",
                           r"<([hH][0-9]+?).*?>.*?</\1>",
                           r"<ins class=(\"|\')(mycorr|authcorr)\1.*?>.*?</ins>",
                           r"<p class=(\"|\')illustration( chapter)?\1>.*?</p>",
                           r"<p class=(\"|\')(ph3|center pfirst|pagenum)\1>.*?</p>",
                           r"(?<=>)CHAPTER.*?(?=<)",
                           r"<p class=(\"|\')(title)\1>.*?\.",
                           r"<!.*?>",
                           r"<b>[0-9]+</b>",
                           r"<p>[ivxIVX]+</p>",
                           r"<p><b>[ivxIVX]+</b>[\s\t\n]*?</p>"
                           ]

    # List of complex expressions to delete surroundings for - in order of precedence
    # list_delete_surround = [r"<span class=\"smcap\">(.*?)</span>"
    #                        ]

    # List of tag expressions to delete matches for - in order of precedence
    list_delete_tags = [r"</?[aAhHpPiIbBuUtTdDrR].*?>",
                        r"</?(small|colgroup|center|strong|span|font|sup|sub|div|br|BR|em|ins|cite|blockquote|pre"
                        r"|li).*?>"
                        ]

    # (?<=x) lookbehind for x
    # (?<!x) negative lookbehind for x
    # Replace mid-sentence line breaks with spaces
    if COUNT:
        log_count += len(re.findall(r"(?<=.)(?<!>)[\r\n]", text_edit))
    text_edit = re.sub(r"(?<=.)(?<!>)[\r\n]", " ", text_edit)

    # Remove all other line breaks
    if COUNT:
        log_count += len(re.findall(r"[\r\n]", text_edit))
    text_edit = re.sub(r"[\r\n]", "", text_edit)

    # Remove complex tags with inner content
    for exp in list_delete_content:
        if COUNT:
            log_count += len(re.findall(exp, text_edit))
        text_edit = re.sub(exp, "", text_edit)

    # Remove complex surrounding tags
    # for exp in list_delete_surround:
    #     if __count__:
    #         count_regex += len(re.findall(exp, line))
    #     line = re.sub(exp, r"\1", line)

    # Class for alphanumeric or punctuation symbol
    c1 = "[A-Za-z0-9_.,:;!? \"\'$]"
    c2 = "[A-Za-z0-9_.,:;!?\"\'$]"

    # Add line breaks at end of tag blocks (<\p> for example)
    # Also removes any number of trailing whitespaces preceding the tag
    if COUNT:
        log_count += len(re.findall(r"[\s\t]*</(p|P)>", text_edit))
    text_edit = re.sub(r"[\s\t]*</(p|P)>", "\n", text_edit)

    # Add line breaks between content blocks
    # if COUNT:
    #     log_count += len(re.findall(rf"(?<={c2})</(p|u|h.|div|span)>(?!{c1}{c2})", text_edit))
    # text_edit = re.sub(rf"(?<={c2})</(p|u|h.|div|span)>(?!{c1}{c2})", r"</\1>\r\n", text_edit)
    # Explanation for this pattern:
    # This pattern matches terminating tags surrounded by alphanumeric/punctual characters.
    # Matches must be preceded by at least one character, and NOT followed by two.
    # The first character after the match must, also, not be a space.

    # Add line breaks at 'break' tags
    # Accounts for <br>, <br/>, and <br /> formats
    if COUNT:
        log_count += len(re.findall(r"<br[\s\t]*/?>", text_edit))
    text_edit = re.sub(r"<br[\s\t]*/?>", "\n", text_edit)

    # Remove all remaining tags
    for exp in list_delete_tags:
        if COUNT:
            log_count += len(re.findall(exp, text_edit))
        text_edit = re.sub(exp, "", text_edit)

    # Remove excessive spaces
    if COUNT:
        log_count += len(re.findall(r"[ ]+", text_edit))
    text_edit = re.sub(r"[ ]+", " ", text_edit)

    # Replace excessive newlines and other whitespace
    # Replaces with a single newline
    if COUNT:
        log_count += len(re.findall(r"\n[\s\n\t]+", text_edit))
    text_edit = re.sub(r"\n[\s\n\t]+", "\n", text_edit)

    # Adds a newline after every '. ' (period with a space after it) (necessary?)
    # if COUNT:
    #     log_count += len(re.findall(r"\. ", text_edit))
    # text_edit = re.sub(r"\. ", ". \n", text_edit)

    # Remove leading whitespace (necessary?)
    # if COUNT:
    #     log_count += len(re.findall(r"\n[\s\n\t]*", text_edit))
    # text_edit = re.sub(r"\n[\s\n\t]*", "\n", text_edit)

    # Report how many unwanted symbols were found after processing
    stranges = len(re.findall(r"[<>]", text_edit))
    print("Found {0} strange symbols!".format(stranges))

    return text_edit


def read_file(usr_fil, log=None):
    """Reads a file and returns each line as a list of strings.

    :param usr_fil: Required - Input file
    :param log: Optional - Log file
    :return: str: The file contents as a string
    """

    out = ""

    try:
        print(usr_fil)
        # Detect file encoding
        file = open(usr_fil, 'rb')
        det = UniversalDetector()
        for line in file.readlines():
            det.feed(line)
            if det.done:
                break
        det.close()
        usr_enc = det.result.get("encoding")

        out += f"File encoding: {usr_enc}\n"

        # Open file using detected encoding
        text_read = open(usr_fil, 'r', encoding='UTF-8', errors='ignore')
        text = text_read.read()
        text_read.close()

        return text
    except IOError:
        out += "\nFile reading failed\n"
        return False
    finally:
        # Log info
        if log is not None:
            log.write(out)
        else:
            print(out, end='')


def clean_file(text_in, f_out, log=None):
    """Cleans input text and sends it to output file.

    :param text_in: Required - Input string list
    :param f_out: Required - Output file
    :param log: Optional - Log file
    :return: bool: Return state
    """

    out = ""

    try:
        log_time = 0
        time_init = 0

        # Time method calls for performance logging
        if TIME:
            time_init = time.perf_counter()
        text_edit = html_edit(text_in)
        text_edit = regex_edit(text_edit)
        if TIME:
            log_time += time.perf_counter() - time_init

        f_out.write(text_edit)
        f_out.close()

        # Debug printout section
        if TIME or COUNT:

            # Stats info
            out += "Stats: "
            if COUNT:
                out += f"{log_count:-7d} edits "
            if COUNT and TIME:
                out += "--"
            if TIME:
                out += f"{log_time:-7.2f} seconds "
            out += "\n"

        if log is not None:
            out += f"Created file: {f_out.name}\n\n"
        print(f"Created file: {f_out.name}\n")
        return True
    except IOError:
        out += "\nText cleaning failed\n"
        return False
    finally:
        # Log info
        if log is not None:
            log.write(out)
        else:
            print(out, end='')


while True:
    print("Options:\n\t1: Clean a file\n\t2: Clean a folder\n\tQ: Quit")
    usr_opt = input("Enter an option: ")
    if usr_opt.lower() == "q":
        break
    elif usr_opt == "1":
        usr_inp = input("Enter a file name: ")
        text = read_file(usr_inp)
        if text:
            p = Path(usr_inp)
            f_out = open(f"{p.parent}/{p.stem}.txt", "w+", encoding='utf-8', errors='replace')
            if not clean_file(text, f_out):
                print("Error while writing file")
        else:
            print("Invalid file")
    elif usr_opt == "2":
        usr_inp = input("Enter a folder name: ")
        log = open(f"clean_text_log.txt", "w+", encoding='utf-8')
        try:
            l_dir = os.listdir(usr_inp)
            # p = Path(usr_inp)
            for item in l_dir:
                p = Path(item)
                print("Reading " + item)
                log.write("Reading " + item + "\n")
                text = read_file(usr_inp.rstrip("/") + "/" + item, log)
                if text:
                    if not os.path.isdir(f"{usr_inp}-clean"):
                        os.mkdir(f"{usr_inp}-clean")
                    f_out = open(f"{usr_inp}-clean/{p.stem}.txt", "w+", encoding='utf-8', errors='replace')
                    if not clean_file(text, f_out, log):
                        print("Error while writing file")
                else:
                    print("Invalid file")
        except FileNotFoundError:
            print("Invalid directory")
        finally:
            log.close()
    else:
        print("Invalid input")

# ...
