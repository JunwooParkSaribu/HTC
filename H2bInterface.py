import os.path
import subprocess
import sys
import signal
import PySimpleGUI as sg
import time


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


def get_gui_path():
    """
    Get the top-level folder path
    :return: Path to list of files using the user settings for this file.  Returns folder of this file if not found
    :rtype: str
    """
    # demo_path = sg.user_settings_get_entry('-demos folder-', os.path.dirname(__file__))
    return os.getcwd()


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


# New function
def get_line_number(file_path, string, dupe_lines):
    lmn = 0
    with open(file_path, encoding="utf-8") as f:
        for num, line in enumerate(f, 1):
            if string.strip() == line.strip() and num not in dupe_lines:
                lmn = num
    return lmn


def window_choose_line_to_edit(filename, full_filename, line_num_list, match_list):
    # sg.popup('matches previously found for this file:', filename, line_num_list)
    i = 0
    if len(line_num_list) == 1:
        return full_filename, line_num_list[0]
    layout = [[sg.T(f'Choose line from {filename}', font='_ 14')]]
    for line in sorted(set(line_num_list)):
        match_text = match_list[i]
        layout += [[sg.Text(f'Line {line} : {match_text}', key=('-T-', line), enable_events=True,
                            size=(min(len(match_text), 90), None))]]
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


def settings_window(path, model_path):
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

    layout = [[sg.T('Settings', font='DEFAULT 25')],
              [sg.T('Change root directory', font='_ 16')],
              [sg.Combo([path],
                        default_value=path, size=(50, 1),
                        key='-FOLDERNAME-'),
               sg.FolderBrowse('Folder Browse', target='-FOLDERNAME-')],
              [sg.T('Select model', font='_ 16')],
              [sg.Combo([model_path],
                        default_value=model_path, size=(50, 1),
                        key='-MODELPATH-'),
               sg.FolderBrowse('Folder Browse', target='-MODELPATH-')],
              [sg.B('Ok', bind_return_key=True), sg.B('Cancel')]
              ]

    window = sg.Window('Settings', layout)

    settings_changed = False

    while True:
        event, values = window.read()
        if event in ('Cancel', sg.WIN_CLOSED):
            break
        if event == 'Ok':
            # sg.user_settings_set_entry('-theme-', values['-THEME-'])
            if path != values['-FOLDERNAME-']:
                settings_changed = True
            break
        elif event == 'Clear History':
            break

    window.close()
    return settings_changed, values['-FOLDERNAME-'], values['-MODELPATH-']


# --------------------------------- Create the window ---------------------------------
def make_window(treedata, starting_path=''):
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

    cutoff_tooltip = "Minimum number of trajectory for a single h2b(default=10)."
    save_tooltip = "Save directory of report files"

    left_col = sg.Column([
        [sg.Tree(data=treedata, headings=[], auto_size_columns=True, num_rows=40, col0_width=80,
                 vertical_scroll_only=False,
                 key='-DEMO LIST-', show_expanded=False, font=("Arial", 13))],
        [sg.Text('Cutoff:', tooltip=cutoff_tooltip),
         sg.Input(size=(10, 1), focus=True, enable_events=True, key='-CUTOFF-', tooltip=cutoff_tooltip),
         sg.CB('Separated report', default=True, enable_events=True, k='-SEPARATE-',
               tooltip='Generate a report file for each trajectory file,'
                       ' Uncheck this box if you want to generate all data into a single report.'),
         sg.CB('Make image', default=True, enable_events=True, k='-MAKEIMAGE-',
               tooltip='Make nucleus trajectory map.'
                       ' if separated report is checked, image will be created for each report.')
         ],
        [sg.Button('Run'),
         # sg.Button('Stop'),
         # sg.Button('Continue'),
         sg.Button('Kill')
         ],
        [sg.Text('Save folder:', tooltip=save_tooltip),
         sg.Text(size=(95, 1), key='-REPORTPATH-'), sg.Button('Browse', tooltip=save_tooltip),
         ]],
        element_justification='l', expand_x=True, expand_y=True)

    right_col = [
        [sg.Multiline(size=(70, 21), write_only=True, expand_x=True, expand_y=True,
                      key='-ML-', reroute_stdout=True, echo_stdout_stderr=True, reroute_cprint=True,
                      autoscroll=True)],
        [sg.B('Settings'), sg.Button('Exit')],
        [sg.T('Python ver ' + sys.version, font='Default 8', pad=(0, 0))],
        [sg.T('Interpreter ' + sg.execute_py_get_interpreter(), font='Default 8', pad=(0, 0))],
    ]
    """
    options_at_bottom = sg.pin(sg.Column([[sg.CB('Evaluation mode', default=False, enable_events=True, k='-EVALMODE-'),
                                           sg.CB('Show ALL file types', default=False, enable_events=True, k='-SHOW ALL FILES-')]],
                                         pad=(0, 0), k='-OPTIONS BOTTOM-', expand_x=True, expand_y=False), expand_x=True, expand_y=False)
    """

    choose_folder_at_top = sg.pin(sg.Column([[sg.T('Current working directory'),
                                              sg.Combo(sorted(sg.user_settings_get_entry('-folder names-', [])),
                                                       default_value=starting_path,
                                                       size=(70, 30), key='-FOLDERNAME-', enable_events=True,
                                                       readonly=True)]], pad=(0, 0), k='-FOLDER CHOOSE-'))
    # ----- Full layout -----
    layout = [[sg.Text('H2B Trajectory Classifier', font='Any 20')],
              [choose_folder_at_top],
              [sg.Pane([sg.Column([[left_col]], element_justification='l', expand_x=True, expand_y=True),
                        sg.Column(right_col, element_justification='c', expand_x=True, expand_y=True)], orientation='h',
                       relief=sg.RELIEF_SUNKEN, expand_x=True, expand_y=True,
                       k='-PANE-')],
              [  # options_at_bottom,
                  sg.Sizegrip()]]

    # --------------------------------- Create Window ---------------------------------
    window = sg.Window('H2B Trajectory Classifier', layout, finalize=True, resizable=True, use_default_focus=False,
                       right_click_menu=sg.MENU_RIGHT_CLICK_EDITME_VER_EXIT)
    window.set_min_size(window.size)
    window.bind('<F1>', '-FOCUS FILTER-')
    window.bind('<F2>', '-FOCUS FIND-')
    window.bind('<F3>', '-FOCUS RE FIND-')
    if not advanced_mode():
        window['-FOLDER CHOOSE-'].update(visible=False)
        window['-RE COL-'].update(visible=False)
        window['-OPTIONS BOTTOM-'].update(visible=False)
    window.bring_to_front()
    return window


def run_command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process


def model_check(model_path) -> bool:
    try:
        isVariables = os.path.isfile(f'{model_path}/variables/variables.data-00000-of-00001')
        isVarIndex = os.path.isfile(f'{model_path}/variables/variables.index')
        isModel = os.path.isfile(f'{model_path}/saved_model.pb')
        if isVariables and isVarIndex and isModel:
            return True
        else:
            sg.cprint(f'Problems on the model, please recheck the model',
                      text_color='white', background_color='red')
            return False
    except:
        sg.cprint(f'Problems on the model, please recheck the model',
                  text_color='white', background_color='red')
        return False


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

    STARTING_PATH = sg.PopupGetFolder('Choose working directory')
    reloaded_dir = STARTING_PATH
    model_dir = f'{get_gui_path()}/model/model_h2b'
    treedata = sg.TreeData()
    add_files_in_folder(treedata, '', STARTING_PATH)
    window = make_window(treedata, STARTING_PATH)
    window.force_focus()
    save_dir = STARTING_PATH
    window['-REPORTPATH-'].update(save_dir)
    window['-CUTOFF-'].update('10')
    data_list = set()
    proc = 0
    stop_status = 0
    while True:
        event, values = window.read(timeout=10000)
        if stop_status == 0:
            if proc != 0:
                cur_time = time.time()
                if proc.poll() == 0:  # End of subprocess
                    window['-ML-'].update('')
                    sg.cprint('Classification on below files...', c='white on green', end='\n')
                    for fichier in file_run_list:
                        sg.cprint(fichier, text_color='white', background_color='purple')
                    sg.cprint(f'Processing is finished', text_color='white', background_color='red')
                    for out in iter(proc.stdout.readline, b''):
                        out = str(out).strip().split("\'")[1].split('\\n')[0]
                        if out.endswith('%'):
                            continue
                        else:
                            sg.cprint(out)
                    sg.cprint(f'Total process time ~ {int(cur_time - start_time)} seconds',
                              text_color='white', background_color='red')
                    start_time = 0
                    proc.kill()
                    proc = 0
                    window.refresh()

                elif proc.poll() is None:
                    window['-ML-'].update('')
                    sg.cprint('Classification on below files...', c='white on green', end='\n')
                    for fichier in file_run_list:
                        sg.cprint(fichier, text_color='white', background_color='purple')
                    sg.cprint(f'Prediction is running...', text_color='white', background_color='red')
                    sg.cprint(f'{int(cur_time - start_time)} seconds')
                    window.refresh()

                else:  # Unexpected subprocess termination handling
                    for out in iter(proc.stdout.readline, b''):
                        out = str(out).strip().split("\'")[1].split('\\n')[0]
                        print(out)
                    print(proc.poll(), proc)
                    proc.kill()
                    proc = 0
                    start_time = 0

        if event in (sg.WINDOW_CLOSED, 'Exit'):
            if proc != 0:
                proc.kill()
            break

        if event == 'Kill':
            if proc != 0:
                start_time = 0
                sg.cprint(f'Subprocess killed : {proc}')
                sg.cprint(f'Terminate prediction', text_color='white', background_color='red')
                proc.stdout.flush()
                proc.kill()
                proc = 0
                window.refresh()
            else:
                sg.cprint(f'Nu current running process')

        elif event == 'Stop':
            sg.cprint(f'Stop prediction', text_color='white', background_color='red')
            if isinstance(proc, int):
                sg.cprint(f'no current running prediction')
            else:
                proc.send_signal(signal.SIGSTOP)
                stop_status = 1

        elif event == 'Continue':
            if proc != 0:
                sg.cprint(f'Continue prediction', text_color='white', background_color='red')
                proc.send_signal(signal.SIGCONT)
                stop_status = 0
            else:
                sg.cprint(f'no current running subprocess')

        elif event == 'Browse':
            save_dir = sg.PopupGetFolder('Report save directory')
            window['-REPORTPATH-'].update(save_dir)

        elif event == 'Run':
            #  model check
            if model_check(model_dir):
                sg.cprint(f'Selected model:{model_dir}',
                          text_color='white', background_color='red')
            else:
                sg.cprint(f'Current selected model:{model_dir}',
                          text_color='white', background_color='red')
                continue

            #  data check
            for selected_data in values['-DEMO LIST-']:
                if '.trxyt' in selected_data.split('/')[-1]:
                    data_list.add(selected_data)
            if len(data_list) == 0:
                sg.cprint(f'There is no selected data, choose trxyt files',
                          text_color='white', background_color='red')
                continue

            if proc != 0:
                sg.cprint(f'Already a prediction is on processing...\n'
                          f'Wait for its end or kill it before starting new one.',
                          text_color='white', background_color='red')
                continue

            stop_status = 0
            cutoff_val = window['-CUTOFF-'].get().strip()
            if len(cutoff_val) == 0:
                cutoff_val = '5'
            if window['-SEPARATE-'].get():
                all_val = 'False'
            else:
                all_val = 'True'
            if window['-MAKEIMAGE-'].get():
                makeimage = 'True'
            else:
                makeimage = 'False'

            file_run_list = []
            for file in data_list:
                file_run_list.append(file)
            with open('config.txt', 'w') as f:
                input_str = ''
                for ficher in file_run_list:
                    input_str += 'data = '
                    input_str += ficher
                    input_str += '\n'
                input_str += f'save_dir = {save_dir}\n'
                input_str += f'model_dir = {model_dir}\n'
                input_str += f'cut_off = {cutoff_val}\n'
                input_str += f'all = {all_val}\n'
                input_str += f'makeImage = {makeimage}\n'

                input_str += '\n'
                input_str += 'immobile_cutoff = 5\n'
                input_str += 'hybrid_cutoff = 12\n'
                input_str += 'amp = 2\n'
                input_str += 'nChannel = 3\n'
                input_str += 'batch_size = 16\n'
                input_str += 'group_size = 160\n'
                f.write(input_str)

            sg.cprint('Classification on below files...', c='white on green', end='\n')
            for fichier in file_run_list:
                sg.cprint(fichier, text_color='white', background_color='purple')
            try:
                # Subprocess calling
                proc = run_command(['python3', 'HTCclassifier.py'])
                start_time = time.time()

            except Exception as e:
                sg.cprint(f'Error trying to run file.  Error info:', e, c='white on red')
            try:
                sg.cprint(f'Prediction is running...', text_color='white', background_color='red')
                #sg.cprint(f'Subprocess created : {proc}')
            except AttributeError:
                sg.cprint('Your version of PySimpleGUI needs to be upgraded to fully use the "WAIT" feature.',
                          c='white on red')

        elif event == 'Settings':
            set_changed, reloaded_dir, model_dir = settings_window(reloaded_dir, model_dir)
            if set_changed is True:
                window.close()
                treedata = sg.TreeData()
                add_files_in_folder(treedata, '', reloaded_dir)
                window = make_window(treedata, reloaded_dir)
                window.force_focus()
                window['-REPORTPATH-'].update(save_dir)
                window['-CUTOFF-'].update('5')

        elif event == '-SHOW ALL FILES-':
            """
            file_list = get_file_list()
            window['-DEMO LIST-'].update(values=file_list)
            window['-FILTER NUMBER-'].update(f'{len(file_list)} files')
            window['-FIND-'].update('')
            """
            pass
    window.close()


if __name__ == '__main__':
    main()
