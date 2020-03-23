# This program is intended to clean html files extracted from the Gutenberg online library.
# Some manual processing, particularly at the beginning and end of the file, will be necessary.
# Otherwise, this program will remove page numbers, images, redundant line breaks, and other markup elements.

# Instructions on use:
# Mode 1: Clean a file
# -- This mode will accept htm/html file names and create a cleaned text file in the same directory as the source file.
# Mode 2: Clean a folder
# -- This mode will accept folder names, read all files in the submitted folder, and create clean versions in a
#   separate folder

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
def regex_edit(line):
    global log_count

    # List of complex expressions to delete matches for - in order of precedence
    list_delete_content = [r"<p><span class=(\"|\')pagenum\1>.*?</span></p>",
                           r"<span class=(\"|\')(pagenum)\1>.*?</span>",
                           r"<p class=(\"|\')(ph2|center pfirst)\1.*?>.*?</p>",
                           r"<(h[0-9]+?).*?>.*?</\1>",
                           r"<ins class=(\"|\')(mycorr|authcorr)\1.*?>.*?</ins>",
                           r"<p class=(\"|\')illustration( chapter)?\1>.*?</p>",
                           r"<p class=(\"|\')(ph3|center pfirst)\1>.*?</p>",
                           r"(?<=>)CHAPTER.*?(?=<)",
                           r"<p class=(\"|\')(title)\1>.*?\.",
                           r"<(a).*?>.*?</\1>"
                           ]

    # List of complex expressions to delete surroundings for - in order of precedence
    # list_delete_surround = [r"<span class=\"smcap\">(.*?)</span>"
    #                        ]

    # List of tag expressions to delete matches for - in order of precedence
    list_delete_tags = [r"</?[ahpibutdr].*?>",
                        r"</?(small|strong|span|font|sup|div|br|em|ins|cite|blockquote).*?>"
                        ]

    # (?<=x) lookbehind for x
    # (?<!x) negative lookbehind for x
    # Replace mid-sentence line breaks with spaces
    if COUNT:
        log_count += len(re.findall(r"(?<=.)(?<!>)[\r\n]", line))
    line = re.sub(r"(?<=.)(?<!>)[\r\n]", " ", line)

    # Remove all other line breaks
    if COUNT:
        log_count += len(re.findall(r"[\r\n]", line))
    line = re.sub(r"[\r\n]", "", line)

    # Remove complex tags with inner content
    for exp in list_delete_content:
        if COUNT:
            log_count += len(re.findall(exp, line))
        line = re.sub(exp, "", line)

    # Remove complex surrounding tags
    # for exp in list_delete_surround:
    #     if __count__:
    #         count_regex += len(re.findall(exp, line))
    #     line = re.sub(exp, r"\1", line)

    # Class for alphanumeric or punctuation symbol
    c1 = "[A-Za-z0-9_.,:;!? \"\'$]"
    c2 = "[A-Za-z0-9_.,:;!?\"\'$]"

    # Add line breaks between content blocks
    if COUNT:
        log_count += len(re.findall(rf"(?<={c2})</(p|u|h.|div|span)>(?!{c1}{c2})", line))
    line = re.sub(rf"(?<={c2})</(p|u|h.|div|span)>(?!{c1}{c2})", r"</\1>\r\n", line)
    # Explanation for this pattern:
    # This pattern matches terminating tags surrounded by alphanumeric/punctual characters.
    # Matches must be preceded by at least one character, and NOT followed by two.
    # The first character after the match must, also, not be a space.

    # Add line breaks at 'break' tags
    if COUNT:
        log_count += len(re.findall(r"<br/>", line))
    line = re.sub(r"<br/>", "\n", line)

    # Remove all remaining tags
    for exp in list_delete_tags:
        if COUNT:
            log_count += len(re.findall(exp, line))
        line = re.sub(exp, "", line)
    return line


def read_file(usr_fil, log=None):
    """Reads a file and returns each line as a list of strings.

    :param usr_fil: Required - Input file
    :param log: Optional - Log file
    :return: list: A list of strings
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
        text_read = open(usr_fil, 'r', encoding=usr_enc, errors='ignore')
        text = text_read.readlines()
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

        line_count = 0
        for line in text_in:
            line_count += 1

            # Time method calls for performance logging
            if TIME:
                time_init = time.perf_counter()
            line = html_edit(line)
            line = regex_edit(line)
            if TIME:
                log_time += time.perf_counter() - time_init

            f_out.write(line)
        f_out.close()

        print(f"Processed {line_count} lines.")
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
