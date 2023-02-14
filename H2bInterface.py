import os.path
import subprocess
import sys
import asyncio
import mmap, re
import signal
import warnings
import PySimpleGUI as sg


def get_file_list_dict():
    """
    Returns dictionary of files
    Key is short filename
    Value is the full filename and path

    :return: Dictionary of demo files
    :rtype: Dict[str:str]
    """

    demo_path = get_demo_path()
    demo_files_dict = {}
    for dirname, dirnames, filenames in os.walk(demo_path):
        for filename in filenames:
            if filename.endswith('.trxyt'):
                fname_full = os.path.join(dirname, filename)
                if filename not in demo_files_dict.keys():
                    demo_files_dict[fname_full] = fname_full
                else:
                    # Allow up to 100 dupicated names. After that, give up
                    print("Duplicated names warnings")
                    for i in range(1, 100):
                        new_filename = f'{filename}_{i}'
                        if new_filename not in demo_files_dict:
                            demo_files_dict[new_filename] = fname_full
                            break

    return demo_files_dict


def get_file_list():
    """
    Returns list of filenames of files to display
    No path is shown, only the short filename

    :return: List of filenames
    :rtype: List[str]
    """
    return sorted(list(get_file_list_dict().keys()))


def add_files_in_folder(treedata, parent, dirname):
    files = os.listdir(dirname)
    for f in files:
        fullname = os.path.join(dirname, f)
        if os.path.isdir(fullname):
            treedata.Insert(parent, fullname, f, values=[])
            add_files_in_folder(treedata, fullname, fullname)
        else:
            if str(f).endswith('.trxyt'):
                treedata.Insert(parent, fullname, f, values=[])


def get_demo_path():
    """
    Get the top-level folder path
    :return: Path to list of files using the user settings for this file.  Returns folder of this file if not found
    :rtype: str
    """
    demo_path = sg.user_settings_get_entry('-demos folder-', os.path.dirname(__file__))

    return '.'


def get_global_editor():
    """
    Get the path to the editor based on user settings or on PySimpleGUI's global settings

    :return: Path to the editor
    :rtype: str
    """
    try:  # in case running with old version of PySimpleGUI that doesn't have a global PSG settings path
        global_editor = sg.pysimplegui_user_settings.get('-editor program-')
    except:
        global_editor = ''
    return global_editor


def get_editor():
    """
    Get the path to the editor based on user settings or on PySimpleGUI's global settings

    :return: Path to the editor
    :rtype: str
    """
    try:  # in case running with old version of PySimpleGUI that doesn't have a global PSG settings path
        global_editor = sg.pysimplegui_user_settings.get('-editor program-')
    except:
        global_editor = ''
    user_editor = sg.user_settings_get_entry('-editor program-', '')
    if user_editor == '':
        user_editor = global_editor

    return user_editor


def using_local_editor():
    user_editor = sg.user_settings_get_entry('-editor program-', None)
    return get_editor() == user_editor


def get_explorer():
    """
    Get the path to the file explorer program

    :return: Path to the file explorer EXE
    :rtype: str
    """
    try:  # in case running with old version of PySimpleGUI that doesn't have a global PSG settings path
        global_explorer = sg.pysimplegui_user_settings.get('-explorer program-', '')
    except:
        global_explorer = ''
    explorer = sg.user_settings_get_entry('-explorer program-', '')
    if explorer == '':
        explorer = global_explorer
    return explorer


def advanced_mode():
    """
    Returns True is advanced GUI should be shown

    :return: True if user indicated wants the advanced GUI to be shown (set in the settings window)
    :rtype: bool
    """
    return sg.user_settings_get_entry('-advanced mode-', True)


def get_theme():
    """
    Get the theme to use for the program
    Value is in this program's user settings. If none set, then use PySimpleGUI's global default theme
    :return: The theme
    :rtype: str
    """
    # First get the current global theme for PySimpleGUI to use if none has been set for this program
    try:
        global_theme = sg.theme_global()
    except:
        global_theme = sg.theme()
    # Get theme from user settings for this program.  Use global theme if no entry found
    user_theme = sg.user_settings_get_entry('-theme-', '')
    if user_theme == '':
        user_theme = global_theme
    return user_theme


# We handle our code properly. But in case the user types in a flag, the flags are now in the middle of a regex. Ignore this warning.

warnings.filterwarnings("ignore", category=DeprecationWarning)


# New function
def get_line_number(file_path, string, dupe_lines):
    lmn = 0
    with open(file_path, encoding="utf-8") as f:
        for num, line in enumerate(f, 1):
            if string.strip() == line.strip() and num not in dupe_lines:
                lmn = num
    return lmn


def kill_ascii(s):
    return "".join([x if ord(x) < 128 else '?' for x in s])


def find_in_file(string, demo_files_dict, regex=False, verbose=False, window=None, ignore_case=True, show_first_match=True):
    """
    Search through the demo files for a string.
    The case of the string and the file contents are ignored

    :param string: String to search for
    :param verbose: if True print the FIRST match
    :type verbose: bool
    :param find_all_matches: if True, then return all matches in the dictionary
    :type find_all_matches: bool
    :return: List of files containing the string
    :rtype: List[str]
    """

    # So you face a predicament here. You wish to read files, both small and large; however the bigger the file/bigger the list, the longer to read the file.
    # This probably isn't what you want, right?
    # Well, we can't use a direct command line to run grep and parse. But it is an option. The user may not have it.
    # We could check if grep exists and if not use our method; but it isn't the best way.
    # So using background knowldge, we know that grep is *very* fast.
    #
    # Why?
    # Grep reads a *ton* of files into memory then searches through the memory to find the string or regex/pattern corresponding to the file.
    # (This is useful if you ever accidently delete a file, grep may be able to get you the contents of it again!)
    # How can we load a file into memory on python as fast as grep whilst keeping it universal?
    # memory mapping (mmap).
    # We can't load a lot of files into memory as we may face issues with watchdog on other operating systems. So we load one file at a time and search though there.
    # This will allow the fastest searching and loading of a file without sacrificing read times.
    # 2.8 seconds on the highend for both small and large files in memory.
    # We also don't have to iterate over lines this way.
    file_list = []
    num_files = 0
    matched_dict = {}
    for file in demo_files_dict:
        try:
            full_filename = demo_files_dict[file]
            if not demo_files_dict == get_file_list_dict():
                full_filename = full_filename[0]
            matches = None

            with open(full_filename, 'rb', 0) as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as s:
                if regex:
                    window['-FIND NUMBER-'].update(f'{num_files} files')
                    window.refresh()
                    matches = re.finditer(bytes("^.*(" + string + ").*$", 'utf-8'), s, re.MULTILINE)
                    if matches:
                        for match in matches:
                            if match is not None:
                                if file not in file_list:
                                    file_list.append(file)
                                    num_files += 1
                                if verbose:
                                    sg.cprint(f"{file}:", c='white on green')
                                    sg.cprint(f"{match.group(0).decode('utf-8')}\n")
                else:
                    window['-FIND NUMBER-'].update(f'{num_files} files')
                    window.refresh()
                    matches = None
                    if ignore_case:
                        if show_first_match:
                            matches = re.search(br'(?i)^' + bytes(".*(" + re.escape(string.lower()) + ").*$", 'utf-8'), s, re.MULTILINE)
                        else:
                            matches = re.finditer(br'(?i)^' + bytes(".*(" + re.escape(string.lower()) + ").*$", 'utf-8'), s, re.MULTILINE)
                    else:
                        if show_first_match:
                            matches = re.search(br'^' + bytes(".*(" + re.escape(string) + ").*$", 'utf-8'), s, re.MULTILINE)
                        else:
                            matches = re.finditer(br'^' + bytes(".*(" + re.escape(string) + ").*$", 'utf-8'), s, re.MULTILINE)
                    if matches:
                        if show_first_match:
                            # file_list.append(file)
                            # num_files += 1
                            match_array = []

                            matched_str = matches.group(0).decode('utf-8')
                            if not all(x in matched_str for x in ("b'", '=')) and len(matched_str) < 500:
                                # safe to assume this is not a base64 string as it does not contain the proper ending
                                match_array.append(matches.group(0).decode('utf-8'))
                                matched_dict[full_filename] = match_array
                                file_list.append(file)
                                num_files += 1
                        else:
                            # We need to do this because strings are "falsy" in Python, but empty matches still return True...
                            append_file = False
                            match_array = []
                            for match_ in matches:
                                matched_str = match_.group(0).decode('utf-8')
                                if matched_str:
                                    if not all(x in matched_str for x in ("b'", '=')) and len(matched_str) < 500:
                                        # if len(match_str) < 500 and "=" not in match_str and "b'" not in match_str:
                                        match_array.append(matched_str)
                                        append_file = True
                            if append_file:
                                file_list.append(file)
                                num_files += 1
                                matched_dict[full_filename] = match_array

                # del matches
        except ValueError:
            del matches
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f'{file}', e, file=sys.stderr)

    # Format of the matches dictionary
    # Filename, [num1, num2, num3]
    file_lines_dict = {}
    list_of_matches = []
    if not regex:
        for key in matched_dict:
            head, tail = os.path.split(key)
            # Tails. Don't wanna put Washington in places he doesn't want to be.
            file_array_old = [key]

            file_array_new = []

            file_match_list = []

            if verbose:
                sg.cprint(f"{tail}:", c='white on green')
            try:
                dupe_lines = []
                for _match in matched_dict[key]:
                    line_num_match = get_line_number(key, _match, dupe_lines)
                    dupe_lines.append(line_num_match)
                    file_array_new.append(line_num_match)
                    file_match_list.append(_match)  # I *really* overthinked this.
                    if verbose:
                        sg.cprint(f"Line: {line_num_match} ", c='white on purple', end='')
                        sg.cprint(f"{_match.strip()}\n")
                    # Make a list of the matches found in this file to add to the dictionry
                    list_of_matches.append(_match.strip())
                file_array_old.append(file_array_new)
                file_array_old.append(file_match_list)

                if tail in file_lines_dict:
                    for i in range(1, 100):
                        new_tail = f'{tail}_{i}'
                        if new_tail not in file_lines_dict:
                            file_lines_dict[new_tail] = file_array_old
                            break
                else:
                    file_lines_dict[tail] = file_array_old
            except Exception as e:
                pass
        find_in_file.file_list_dict = file_lines_dict

    file_list = list(set(file_list))
    return file_list


def window_choose_line_to_edit(filename, full_filename, line_num_list, match_list):
    # sg.popup('matches previously found for this file:', filename, line_num_list)
    i = 0
    if len(line_num_list) == 1:
        return full_filename, line_num_list[0]
    layout = [[sg.T(f'Choose line from {filename}', font='_ 14')]]
    for line in sorted(set(line_num_list)):
        match_text = match_list[i]
        layout += [[sg.Text(f'Line {line} : {match_text}', key=('-T-', line), enable_events=True, size=(min(len(match_text), 90), None))]]
        i += 1
    layout += [[sg.B('Cancel')]]

    window = sg.Window('Open Editor', layout)

    line_chosen = line_num_list[0]
    while True:
        event, values = window.read()
        if event in ('Cancel', sg.WIN_CLOSED):
            line_chosen = None
            break
        # At this point we know a line was chosen
        line_chosen = event[1]
        break

    window.close()
    return full_filename, line_chosen


def settings_window():
    """
    Show the settings window.
    This is where the folder paths and program paths are set.
    Returns True if settings were changed

    :return: True if settings were changed
    :rtype: (bool)
    """

    try:
        global_editor = sg.pysimplegui_user_settings.get('-editor program-')
    except:
        global_editor = ''
    try:
        global_explorer = sg.pysimplegui_user_settings.get('-explorer program-')
    except:
        global_explorer = ''
    try:  # in case running with old version of PySimpleGUI that doesn't have a global PSG settings path
        global_theme = sg.theme_global()
    except:
        global_theme = ''

    layout = [[sg.T('Program Settings', font='DEFAULT 25')],
              [sg.T('Path to Tree', font='_ 16')],
              [sg.Combo(sorted(sg.user_settings_get_entry('-folder names-', [])), default_value=sg.user_settings_get_entry('-demos folder-', get_demo_path()), size=(50, 1),
                        key='-FOLDERNAME-'),
               sg.FolderBrowse('Folder Browse', target='-FOLDERNAME-'), sg.B('Clear History')],
              [sg.T('Editor Program', font='_ 16')],
              [sg.T('Leave blank to use global default'), sg.T(global_editor)],
              [sg.In(sg.user_settings_get_entry('-editor program-', ''), k='-EDITOR PROGRAM-'), sg.FileBrowse()],
              [sg.T('File Explorer Program', font='_ 16')],
              [sg.T('Leave blank to use global default'), sg.T(global_explorer)],
              [sg.In(sg.user_settings_get_entry('-explorer program-'), k='-EXPLORER PROGRAM-'), sg.FileBrowse()],
              [sg.T('Theme', font='_ 16')],
              [sg.T('Leave blank to use global default'), sg.T(global_theme)],
              [sg.Combo([''] + sg.theme_list(), sg.user_settings_get_entry('-theme-', ''), readonly=True, k='-THEME-')],
              [sg.T('Double-click a File Will:'), sg.R('Run', 2, sg.user_settings_get_entry('-dclick runs-', False), k='-DCLICK RUNS-'),
               sg.R('Edit', 2, sg.user_settings_get_entry('-dclick edits-', False), k='-DCLICK EDITS-'),
               sg.R('Nothing', 2, sg.user_settings_get_entry('-dclick none-', False), k='-DCLICK NONE-')],
              [sg.CB('Use Advanced Interface', default=advanced_mode(), k='-ADVANCED MODE-')],
              [sg.B('Ok', bind_return_key=True), sg.B('Cancel')],
              ]

    window = sg.Window('Settings', layout)

    settings_changed = False

    while True:
        event, values = window.read()
        if event in ('Cancel', sg.WIN_CLOSED):
            break
        if event == 'Ok':
            sg.user_settings_set_entry('-folder names-', [])
            sg.user_settings_set_entry('-last filename-', '')
            sg.user_settings_set_entry('-demos folder-', values['-FOLDERNAME-'])
            print('pressed ok in settings',values['-FOLDERNAME-'])
            sg.user_settings_set_entry('-editor program-', values['-EDITOR PROGRAM-'])
            sg.user_settings_set_entry('-theme-', values['-THEME-'])
            sg.user_settings_set_entry('-folder names-', list(set(sg.user_settings_get_entry('-folder names-', []) + [values['-FOLDERNAME-'], ])))
            sg.user_settings_set_entry('-explorer program-', values['-EXPLORER PROGRAM-'])
            sg.user_settings_set_entry('-advanced mode-', values['-ADVANCED MODE-'])
            sg.user_settings_set_entry('-dclick runs-', values['-DCLICK RUNS-'])
            sg.user_settings_set_entry('-dclick edits-', values['-DCLICK EDITS-'])
            sg.user_settings_set_entry('-dclick nothing-', values['-DCLICK NONE-'])
            settings_changed = True
            break
        elif event == 'Clear History':
            break

    window.close()
    return settings_changed


ML_KEY = '-ML-'  # Multline's key


# --------------------------------- Create the window ---------------------------------
def make_window(treedata):
    """
    Creates the main window
    :return: The main window object
    :rtype: (sg.Window)
    """

    theme = get_theme()
    if not theme:
        theme = sg.OFFICIAL_PYSIMPLEGUI_THEME
    sg.theme(theme)
    # First the window layout...2 columns

    cutoff_tooltip = "Minimum number of h2b trajectory(default=5)."
    save_tooltip = "Report file save directory"


    left_col = sg.Column([
        #[sg.Listbox(values=get_file_list(), select_mode=sg.SELECT_MODE_EXTENDED, size=(50, 50), bind_return_key=True, key='-DEMO LIST-', expand_x=True, expand_y=True)],
        [sg.Tree(data=treedata, headings=[], auto_size_columns=True, num_rows=40, col0_width=90,
                 key='-DEMO LIST-', show_expanded=False),
         ],
        [sg.Text('Cutoff:', tooltip=cutoff_tooltip), sg.Input(size=(10, 1), focus=True, enable_events=True, key='-CUTOFF-', tooltip=cutoff_tooltip),
         sg.CB('Seperate report', enable_events=True, k='-SEPERATE-', tooltip='Generate a report file for each trajectory file')],
        [sg.Button('Run'), sg.Button('Stop'), sg.Button('Continue'), sg.Button('Kill')],
        [sg.Text('Save folder:', tooltip=save_tooltip), sg.Button('Browse'), sg.Text('current save path:'), sg.Text(size=(70,1), key='-REPORTPATH-')]
    ], element_justification='l', expand_x=True, expand_y=True)


    right_col = [
        [sg.Multiline(size=(70, 21), write_only=True, expand_x=True, expand_y=True, key=ML_KEY, reroute_stdout=True, echo_stdout_stderr=True, reroute_cprint=True)],
        [sg.B('Settings'), sg.Button('Exit')],
        [sg.T('Python ver ' + sys.version, font='Default 8', pad=(0, 0))],
        [sg.T('Interpreter ' + sg.execute_py_get_interpreter(), font='Default 8', pad=(0, 0))],
    ]

    options_at_bottom = sg.pin(sg.Column([[sg.CB('Verbose', enable_events=True, k='-VERBOSE-', tooltip='Enable to see the matches in the right hand column'),
                                           sg.CB('Show only first match in file', default=True, enable_events=True, k='-FIRST MATCH ONLY-',
                                                 tooltip='Disable to see ALL matches found in files'),
                                           sg.CB('Find ignore case', default=True, enable_events=True, k='-IGNORE CASE-'),
                                           sg.CB('Wait for Runs to Complete', default=False, enable_events=True, k='-WAIT-'),
                                           sg.CB('Show ALL file types', default=False, enable_events=True, k='-SHOW ALL FILES-'),
                                           ]],
                                         pad=(0, 0), k='-OPTIONS BOTTOM-', expand_x=True, expand_y=False), expand_x=True, expand_y=False)

    choose_folder_at_top = sg.pin(sg.Column([[sg.T('Click settings to set top of your tree or choose a previously chosen folder'),
                                              sg.Combo(sorted(sg.user_settings_get_entry('-folder names-', [])), default_value=sg.user_settings_get_entry('-demos folder-', ''),
                                                       size=(50, 30), key='-FOLDERNAME-', enable_events=True, readonly=True)]], pad=(0, 0), k='-FOLDER CHOOSE-'))
    # ----- Full layout -----

    layout = [[sg.Text('H2B Trajectory Classifier', font='Any 20')],
              [choose_folder_at_top],
              # [sg.Column([[left_col],[ lef_col_find_re]], element_justification='l',  expand_x=True, expand_y=True), sg.Column(right_col, element_justification='c', expand_x=True, expand_y=True)],
              [sg.Pane([sg.Column([[left_col]], element_justification='l', expand_x=True, expand_y=True),
                        sg.Column(right_col, element_justification='c', expand_x=True, expand_y=True)], orientation='h', relief=sg.RELIEF_SUNKEN, expand_x=True, expand_y=True,
                       k='-PANE-')],
              [options_at_bottom, sg.Sizegrip()]]

    # --------------------------------- Create Window ---------------------------------
    window = sg.Window('H2B Trajectory Classifier', layout, finalize=True, resizable=True, use_default_focus=False, right_click_menu=sg.MENU_RIGHT_CLICK_EDITME_VER_EXIT)
    window.set_min_size(window.size)

    # window.bind("<Alt_L><x>", 'Exit')       # matches the underscore shown on the Exit button (For now disabled this feature until buttons with underscore released to PyPI)

    window.bind('<F1>', '-FOCUS FILTER-')
    window.bind('<F2>', '-FOCUS FIND-')
    window.bind('<F3>', '-FOCUS RE FIND-')
    if not advanced_mode():
        window['-FOLDER CHOOSE-'].update(visible=False)
        window['-RE COL-'].update(visible=False)
        window['-OPTIONS BOTTOM-'].update(visible=False)

    # sg.cprint_set_output_destination(window, ML_KEY)
    window.bring_to_front()
    return window


def run_command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process


# --------------------------------- Main Program Layout ---------------------------------
def main():
    """
    The main program that contains the event loop.
    It will call the make_window function to create the window.
    """
    try:
        version = sg.version
        version_parts = version.split('.')
        major_version, minor_version = int(version_parts[0]), int(version_parts[1])
        if major_version < 4 or (major_version == 4 and minor_version < 32):
            sg.popup('Warning - Your PySimpleGUI version is less then 4.35.0',
                     'As a result, you will not be able to use the EDIT features of this program',
                     'Please upgrade to at least 4.35.0',
                     f'You are currently running version:',
                     sg.version,
                     background_color='red', text_color='white')
    except Exception as e:
        print(f'** Warning Exception parsing version: {version} **  ', f'{e}')
    icon = sg.EMOJI_BASE64_HAPPY_IDEA
    sg.user_settings_filename('psgdemos.json')
    sg.set_options(icon=icon)
    find_in_file.file_list_dict = None

    old_typed_value = None

    STARTING_PATH = sg.PopupGetFolder('Folder to display')
    treedata = sg.TreeData()
    add_files_in_folder(treedata, '', STARTING_PATH)

    file_list_dict = get_file_list_dict()
    file_list = get_file_list()
    window = make_window(treedata)
    window.force_focus()
    counter = 0
    save_dir = '.'
    window['-REPORTPATH-'].update(save_dir)
    window['-CUTOFF-'].update('5')
    proc = 0
    while True:
        event, values = window.read(timeout=10000)
        if proc != 0:
            if proc.stdout != None:
                if proc.poll()==0:
                    outs = str(proc.stdout.read())
                    proc = 0
                    outs = outs.strip().split('\'')[1].split('\\n')
                    for out in outs:
                        print(out)
                # pass if no readline
                else:
                    outs = str(proc.stdout.readline())
                    outs = outs.strip().split('\'')[1].split('\\n')[0]
                    print(outs)
                window.refresh()
        #print(f'event:{event}, {values}')
        counter += 1
        if event in (sg.WINDOW_CLOSED, 'Exit'):
            if proc != 0:
                proc.kill()
            break

        if event == 'Kill':
            if proc != 0:
                if proc.stdout != None:
                    print('Terminate prediction')
                    proc.stdout.flush()
                    proc.kill()
                    proc = 0
                    window.refresh()

        if event == 'Browse':
            save_dir = sg.PopupGetFolder('Report save directory')
            window['-REPORTPATH-'].update(save_dir)

        if event == 'Run':
            cutoff_val = window['-CUTOFF-'].get().strip()
            if len(cutoff_val) == 0:
                cutoff_val = '5'
            if window['-SEPERATE-'].get():
                all_val = 'False'
            else:
                all_val = 'True'


            file_run_list = []
            for file in values['-DEMO LIST-']:
                file_run_list.append(file)
            with open('config.txt', 'w') as f:
                input_str = ''
                for ficher in file_run_list:
                    input_str += 'data = '
                    input_str += ficher
                    input_str += '\n'

                input_str += f'save_dir = {save_dir}\n'
                input_str += 'model_dir = 11_02_10500samples\n'
                input_str += f'cut_off = {cutoff_val}\n'
                input_str += f'all = {all_val}\n'
                input_str += 'amp = 2\n'
                input_str += 'nChannel = 3\n'
                input_str += 'batch_size = 200\n'
                input_str += 'group_size = 2000\n'
                f.write(input_str)

            sg.cprint('Classification on below files...', c='white on green', end='')
            sg.cprint('')
            for fichier in file_run_list:
                sg.cprint(fichier, text_color='white', background_color='purple')
            try:
                # Subprocess calling
                proc = run_command(['python3', 'HTCclassifier.py'])
            except Exception as e:
                sg.cprint(f'Error trying to run file.  Error info:', e, c='white on red')
            try:
                sg.cprint(f'Waiting for results..', text_color='white', background_color='red')

                #if proc != 0:
                #    if proc.stdout != None:
                #        print(proc.stdout.readline())
                """               
                print(p)
                while p==0:
                    line = next(iter(p.stdout.readline, b''))
                    subp_buffer = str(line.strip())
                    print(subp_buffer.split('\'')[1])
                    window.refresh()
                    event, values = window.read(timeout=5000)
                    print(event)
                    if event == 'Stop':
                        print('Stop prediction')
                        p.send_signal(signal.SIGSTOP)
                    if event == 'Continue':
                        print('Continue prediction')
                        p.send_signal(signal.SIGCONT)
                    if event == 'Kill':
                        print('Terminate prediction')
                        p.kill()
                """

            except AttributeError:
                sg.cprint('Your version of PySimpleGUI needs to be upgraded to fully use the "WAIT" feature.', c='white on red')


        elif event.startswith('Edit Me'):
            editor_program = get_editor()
            sg.cprint(f'opening using {editor_program}:')
            sg.cprint(f'{__file__}', text_color='white', background_color='red', end='')
            sg.execute_command_subprocess(f'{editor_program}', f'"{__file__}"')
        elif event == '-FILTER-':
            new_list = [i for i in file_list if values['-FILTER-'].lower() in i.lower()]
            window['-DEMO LIST-'].update(new_list)
            window['-FILTER NUMBER-'].update(f'{len(new_list)} files')
            window['-FIND NUMBER-'].update('')
            window['-FIND-'].update('')
            window['-FIND RE-'].update('')
        elif event == '-FOCUS FIND-':
            window['-FIND-'].set_focus()
        elif event == '-FOCUS FILTER-':
            window['-FILTER-'].set_focus()
        elif event == '-FOCUS RE FIND-':
            window['-FIND RE-'].set_focus()
        elif event == '-FIND-' or event == '-FIRST MATCH ONLY-' or event == '-VERBOSE-' or event == '-FIND RE-':
            is_ignore_case = values['-IGNORE CASE-']
            old_ignore_case = False
            current_typed_value = str(values['-FIND-'])
            if len(values['-FIND-']) == 1:
                window[ML_KEY].update('')
                window['-VERBOSE-'].update(False)
                values['-VERBOSE-'] = False
            if values['-VERBOSE-']:
                window[ML_KEY].update('')
            if values['-FIND-']:
                if find_in_file.file_list_dict is None or old_typed_value is None or old_ignore_case is not is_ignore_case:
                    # New search.
                    old_typed_value = current_typed_value
                    file_list = find_in_file(values['-FIND-'], get_file_list_dict(), verbose=values['-VERBOSE-'], window=window, ignore_case=is_ignore_case,
                                             show_first_match=values['-FIRST MATCH ONLY-'])
                elif current_typed_value.startswith(old_typed_value) and old_ignore_case is is_ignore_case:
                    old_typed_value = current_typed_value
                    file_list = find_in_file(values['-FIND-'], find_in_file.file_list_dict, verbose=values['-VERBOSE-'], window=window, ignore_case=is_ignore_case,
                                             show_first_match=values['-FIRST MATCH ONLY-'])
                else:
                    old_typed_value = current_typed_value
                    file_list = find_in_file(values['-FIND-'], get_file_list_dict(), verbose=values['-VERBOSE-'], window=window, ignore_case=is_ignore_case,
                                             show_first_match=values['-FIRST MATCH ONLY-'])
                window['-DEMO LIST-'].update(sorted(file_list))
                window['-FIND NUMBER-'].update(f'{len(file_list)} files')
                window['-FILTER NUMBER-'].update('')
                window['-FIND RE-'].update('')
                window['-FILTER-'].update('')
            elif values['-FIND RE-']:
                window['-ML-'].update('')
                file_list = find_in_file(values['-FIND RE-'], get_file_list_dict(), regex=True, verbose=values['-VERBOSE-'], window=window)
                window['-DEMO LIST-'].update(sorted(file_list))
                window['-FIND NUMBER-'].update(f'{len(file_list)} files')
                window['-FILTER NUMBER-'].update('')
                window['-FIND-'].update('')
                window['-FILTER-'].update('')
        elif event == 'Find RE':
            window['-ML-'].update('')
            file_list = find_in_file(values['-FIND RE-'], get_file_list_dict(), regex=True, verbose=values['-VERBOSE-'], window=window)
            window['-DEMO LIST-'].update(sorted(file_list))
            window['-FIND NUMBER-'].update(f'{len(file_list)} files')
            window['-FILTER NUMBER-'].update('')
            window['-FIND-'].update('')
            window['-FILTER-'].update('')
            sg.cprint('Regular expression find completed')
        elif event == 'Settings':
            if settings_window() is True:
                reloaded_dir = sg.user_settings_get_entry('-folder names-', [])[0]
                window.close()

                treedata = sg.TreeData()
                add_files_in_folder(treedata, '', reloaded_dir)

                file_list_dict = get_file_list_dict()
                file_list = get_file_list()
                window = make_window(treedata)
                window.force_focus()
                save_dir = '.'
                window['-REPORTPATH-'].update(save_dir)
                window['-CUTOFF-'].update('5')

        elif event == 'Clear':
            file_list = get_file_list()
            window['-FILTER-'].update('')
            window['-FILTER NUMBER-'].update(f'{len(file_list)} files')
            window['-FIND-'].update('')
            window['-DEMO LIST-'].update(file_list)
            window['-FIND NUMBER-'].update('')
            window['-FIND RE-'].update('')
            window['-ML-'].update('')
        elif event == '-FOLDERNAME-':
            sg.user_settings_set_entry('-demos folder-', values['-FOLDERNAME-'])
            file_list_dict = get_file_list_dict()
            file_list = get_file_list()
            window['-DEMO LIST-'].update(values=file_list)
            window['-FILTER NUMBER-'].update(f'{len(file_list)} files')
            window['-ML-'].update('')
            window['-FIND NUMBER-'].update('')
            window['-FIND-'].update('')
            window['-FIND RE-'].update('')
            window['-FILTER-'].update('')
        elif event == 'Open Folder':
            explorer_program = get_explorer()
            if explorer_program:
                sg.cprint(f'Opening Folder using {explorer_program}...', c='white on green', end='')
                sg.cprint('')
                for file in values['-DEMO LIST-']:
                    file_selected = str(file_list_dict[file])
                    file_path = os.path.dirname(file_selected)
                    if sg.running_windows():
                        file_path = file_path.replace('/', '\\')
                    sg.cprint(file_path, text_color='white', background_color='purple')
                    sg.execute_command_subprocess(explorer_program, file_path)
        elif event == 'Copy Path':
            for file in values['-DEMO LIST-']:
                sg.cprint('Copying the last highlighted filename in your list')
                if find_in_file.file_list_dict is not None:
                    full_filename, line = window_choose_line_to_edit(file, find_in_file.file_list_dict[file][0], find_in_file.file_list_dict[file][1],
                                                                     find_in_file.file_list_dict[file][2])
                else:
                    full_filename, line = get_file_list_dict()[file], 1
                if line is not None:
                    sg.cprint(f'Added to Clipboard Full Path {full_filename}', c='white on purple')
                    sg.clipboard_set(full_filename)
        elif event == 'Version':
            sg.popup_scrolled(sg.get_versions(), keep_on_top=True, non_blocking=True)
        elif event == '-SHOW ALL FILES-':
            python_only = not values[event]
            file_list_dict = get_file_list_dict()
            file_list = get_file_list()
            window['-DEMO LIST-'].update(values=file_list)
            window['-FILTER NUMBER-'].update(f'{len(file_list)} files')
            window['-ML-'].update('')
            window['-FIND NUMBER-'].update('')
            window['-FIND-'].update('')
            window['-FIND RE-'].update('')
            window['-FILTER-'].update('')
    window.close()


if __name__ == '__main__':
    main()
