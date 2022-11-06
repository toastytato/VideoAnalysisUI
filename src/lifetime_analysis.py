from analysis_params import Analysis
import filter_params as fp
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage.segmentation import watershed
import numpy as np
import cv2
import imutils
import os
import math
import pandas as pd

from misc_methods import MyFrame, register_my_param


class Bubble:
    id_cnt = 0

    def __init__(self, mask, init_idx) -> None:
        self.id = Bubble.id_cnt
        Bubble.id_cnt += 1

        self.init_mask = mask
        self.x, self.y, self.r = self.get_circle_from_mask(mask)
        self.roi_x1 = int(self.x - self.r)
        self.roi_y1 = int(self.y - self.r)
        self.roi_x2 = int(self.roi_x1 + self.r * 2)
        self.roi_y2 = int(self.roi_y1 + self.r * 2)
        self.cropped_mask = self.init_mask[self.roi_y1:self.roi_y2,
                                           self.roi_x1:self.roi_x2]
        self.lifetime = 1
        self.init_idx = init_idx
        self.exists = True

    def get_circle_from_mask(self, mask):
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        # draw a circle enclosing the object
        # keep r if using min enclosing circle radius
        ((x, y), r) = cv2.minEnclosingCircle(c)
        # get center via Center of Mass
        # M_1 = cv2.moments(c)
        # if M_1['m00'] == 0:
        #     M_1['m00', 'm01'] = 1
        # x = int((M_1['m10'] + .1) / (M_1['m00'] + .1))
        # y = int((M_1['m01'] + .1) / (M_1['m00'] + .1))

        fit_circle = 'perimeter'
        if fit_circle == 'area':
            # 1.5 because the circle looks small
            r = math.sqrt(1.2 * area / math.pi)
        elif fit_circle == 'perimeter':
            r = cv2.arcLength(c, True) / (2 * math.pi)
        return x, y, r

    @property
    def ipos(self):
        return (int(self.x), int(self.y))

    @classmethod
    def reset_id(cls):
        cls.id_cnt = 0

    def check_exists(self, frame, frame_idx):
        frame = frame[self.roi_y1:self.roi_y2, self.roi_x1:self.roi_x2]
        result = cv2.bitwise_and(frame, self.cropped_mask)
        self.exists = np.any(result)
        if not self.exists:
            self.lifetime = frame_idx - self.init_idx


class AnalyzeBubblesLifetime(Analysis):

    def __init__(self, **opts):

        self.dir = "D:\\Research\\Bubble 2 Videos\\Small Trimmed"
        self.video_idx = 30
        self.video_list = [
            f for f in os.listdir(self.dir) if f.endswith('.mp4')
        ]
        self.num_videos = len(self.video_list)
        print(self.video_list)
        opts['url'] = f"{self.dir}\\{self.video_list[self.video_idx]}"

        self.filters = [
            fp.Normalize(),
            fp.Blur(radius=2),
            fp.Threshold(type='otsu'),
            fp.Invert(),
            fp.Dilate(),
            fp.Erode()
        ]

        params = [
            {
                'title': 'Preprocessing',
                'name': 'filter_group',
                'type': 'FilterGroup',
                'expanded': False,
                'children': self.filters
            },
            {
                'title':
                'Analysis Params',
                'name':
                'analysis',
                'type':
                'group',
                'children': [
                    {
                        'title': 'Next Analysis',
                        'name': 'next_analysis',
                        'type': 'action',
                    },
                    {
                        'title': 'Skip Analysis',
                        'name': 'skip_analysis',
                        'type': 'action',
                    },
                    {
                        'title': 'Anaysis State',
                        'name': 'analysis_state',
                        'type': 'int',
                        'value': 1
                    },
                    {
                        'title': 'Peak Region Size',
                        'name': 'peak_footprint',
                        'type': 'slider',
                        'value': 25,
                        'limits': (1, 50),
                    },
                    {
                        'title': 'Debug View',
                        'name': 'debug_view',
                        'type': 'bool',
                        'value': True
                    },
                    {
                        'title': 'View List',
                        'name': 'view_list',
                        'type': 'list',
                        'value': "",
                        'limits': [],  # empty now, fill once at end of analyze
                        'tip': 'View intermediary frames'
                    }
                ]
            },
            {
                'title':
                'Overlay',
                'name':
                'overlay',
                'type':
                'group',
                'children': [
                    {
                        'title': 'Full Scale',
                        'name': 'fs_view',
                        'type': 'bool',
                        'value': True
                    },
                    {
                        'title': 'Colormap',
                        'name': 'colormap',
                        'type': 'bool',
                        'value': False
                    },
                    {
                        'title': 'Show ID',
                        'name': 'show_id',
                        'type': 'bool',
                        'value': False
                    },
                ]
            }
        ]

        # check if children exists in case loading from saved state
        if 'children' not in opts:
            opts['children'] = params

        super().__init__(**opts)

        self.view = {}
        self.view_set = False
        self.export_cnt = 0

        self.bubbles = []

    def analyze(self, frame):
        frame = frame.cvt_color('gray')
        super().analyze(frame)  # sets orig.frame

        analysis_state = self.child('analysis', 'analysis_state').value()

        # take the difference
        if analysis_state == 2:
            self.child('filter_group', 'Normalize').set_normalized()
            self.view['filtered'] = self.crop_to_roi(
                self.child('filter_group').preprocess(frame))
        # set frame index to start of bubble analysis
        elif analysis_state == 3:
            self.view['filtered'] = self.crop_to_roi(
                self.child('filter_group').preprocess(frame))
            self.get_url_info(self.child('settings', 'File Select').value())
            self.curr_frame_idx = self.analysis_start_idx

        elif analysis_state == 4:
            frame = self.child('filter_group').preprocess(frame)
            self.set_auto_roi(frame)
            self.view['filtered'] = self.crop_to_roi(frame)

            self.view['dist'] = ndi.distance_transform_edt(
                self.view['filtered'])
            fp = self.child('analysis', 'peak_footprint').value()
            # take the peak brightness in the distance transforms as that would be around the center of the bubble
            coords = peak_local_max(self.view['dist'],
                                    footprint=np.ones((fp, fp)),
                                    labels=self.view['filtered'])
            self.view['seed'] = np.zeros(self.view['dist'].shape,
                                         dtype=np.uint8)
            self.view['seed'][tuple(coords.T)] = True
            # for seeds that are too close to each other, merge them
            self.view['seed'] = cv2.dilate(self.view['seed'],
                                           kernel=None,
                                           iterations=1)
            markers, _ = ndi.label(self.view['seed'])
            self.view['watershed'] = watershed(-self.view['dist'],
                                               markers,
                                               mask=self.view['filtered'])
            self.bubbles = []
            Bubble.reset_id()
            # ignore 0: background, get all other labels
            for label in np.unique(self.view['watershed'])[1:]:
                mask = np.zeros(self.view['watershed'].shape, dtype='uint8')
                mask[self.view['watershed'] == label] = 255
                self.bubbles.append(Bubble(mask, self.curr_frame_idx))

            self.view_set = False

        elif analysis_state == 5:
            self.child('analysis', 'debug_view').setValue(False)
            self.child('filter_group', 'view_list').setValue('Blur')
            self.child('overlay', 'show_id').setValue(True)

        elif analysis_state == 6:
            self.view['filtered'] = self.crop_to_roi(
                self.child('filter_group').preprocess(frame))
            empty = True
            for b in self.bubbles:
                # if existed in previous frame
                if b.exists:
                    # check if current frame it's still there
                    empty = False
                    b.check_exists(self.view['filtered'], self.curr_frame_idx)
            if empty:
                self.child('analysis', 'analysis_state').setValue(7)
                self.export_data()
                self.is_playing = False
        # on the first run through display all debug frames
        if not self.view_set:
            self.child('analysis',
                       'view_list').setLimits(list(self.view.keys()))
            self.view_set = True

    def annotate(self, frame):
        curr_analysis_state = self.child('analysis', 'analysis_state').value()
        # analysis state edge triggered actions
        if curr_analysis_state < 4:
            if curr_analysis_state == 3:
                self.child('filter_group', 'Blur', 'radius').setValue(2)
            self.child('analysis',
                       'analysis_state').setValue(curr_analysis_state + 1)
            self.request_analysis_update.emit()
            return frame
        # don't skip overlay since we need to save this frame for reference
        elif curr_analysis_state == 5:
            self.child('analysis',
                       'analysis_state').setValue(curr_analysis_state + 1)
            self.request_analysis_update.emit()

        if self.child('analysis', 'debug_view').value() and self.view:
            view_frame = self.view[self.child('analysis', 'view_list').value()]
        else:
            view_frame = self.crop_to_roi(
                self.child('filter_group').get_preview())

        x, y = self.cursor_pos
        if (0 <= y < view_frame.shape[0] and 0 <= x < view_frame.shape[1]):
            self.set_cursor_value(self.cursor_pos, view_frame[y, x])

        if view_frame is not None:
            # don't crop because the images stored in the analysis are already cropped
            frame = MyFrame(view_frame).cvt_color('bgr')
        else:
            frame = self.crop_to_roi(frame.cvt_color('bgr'))

        if self.child('overlay', 'fs_view').value():
            frame = self.fs_stretch(frame)

        if self.child('overlay', 'colormap').value():
            frame = cv2.applyColorMap(np.uint8(frame), cv2.COLORMAP_JET)

        for b in self.bubbles:
            if self.child('overlay', 'show_id').value():
                cv2.putText(frame, str(b.id), (int(b.x) - 11, int(b.y) + 7),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
            if b.exists:
                cv2.circle(frame, b.ipos, int(b.r), (0, 255, 0), 1)
            else:
                cv2.circle(frame, b.ipos, int(b.r), (255, 0, 0), 1)

        if curr_analysis_state == 5:
            self.export_frame(frame)

        return frame

    def on_param_change(self, parameter, changes):
        # super().on_param_change(parameter, changes)
        param, change, data = changes[0]
        parent = param.parent().name()
        name = param.name()

        if name == 'File Select':
            self.child('analysis', 'analysis_state').setValue(1)
            self.child('settings', 'curr_frame_idx').setValue(0)
        elif name == 'next_analysis':
            if self.child('analysis', 'analysis_state').value() == 4:
                self.child('analysis', 'analysis_state').setValue(5)
                self.child('filter_group', 'Threshold',
                           'thresh_type').setValue('thresh')
                self.child('filter_group', 'Blur', 'radius').setValue(1)
                self.curr_frame_idx += 1
                self.is_playing = True
            if (self.child('analysis', 'analysis_state').value() == 7
                    and self.video_idx < self.num_videos):
                self.video_idx += 1
                url = f"{self.dir}\\{self.video_list[self.video_idx]}"
                self.child('settings',
                           'File Select').setValue(url)

        if (parent == 'overlay' or name == 'debug_view'
                or (parent == 'analysis' and name == 'view_list')
                or name == 'cursor_info'):
            self.request_annotate_update.emit()
        else:
            self.request_analysis_update.emit()

    def fs_stretch(self, frame):
        diff = frame.max() - frame.min()
        p = (255 + 1) / (diff + 1)  # + 1 to avoid div by 0
        a = frame.min() * p
        # print(f'p:{p}, a:{a}, min:{frame.min()}, max:{frame.max()}')
        return p * frame - a

    def set_auto_roi(self, frame):
        coords = np.nonzero(frame)
        if len(coords[0]):
            top_y = np.max(coords[0]) + 10
            bot_y = np.min(coords[0]) - 10
            (h, w) = frame.shape[:2]
            self.opts['roi'] = [0, bot_y, w, top_y - bot_y]

    def get_url_info(self, url):
        filename = os.path.basename(url).split('.')[0]
        print(filename)
        self.exposure_time, val1, val2 = filename.split('-')
        self.video_start_idx = int(val1)
        self.video_bubble_idx = int(val2)
        self.analysis_start_idx = self.video_bubble_idx - self.video_start_idx

    def export_data(self):
        if self.export_cnt > 0:
            mode = 'a'
        else:
            mode = 'w'

        with pd.ExcelWriter('lifetime_export/bubble_lifetime.xlsx',
                            mode=mode) as w:
            df = pd.DataFrame()
            df['id'] = [b.id for b in self.bubbles]
            df['x (px)'] = [b.x for b in self.bubbles]
            df['y (px)'] = [b.y for b in self.bubbles]
            df['r (px)'] = [b.r for b in self.bubbles]
            df['lifetime'] = [b.lifetime for b in self.bubbles]
            df = df.sort_values(by=['x (px)'])
            diff = df.diff()
            df['x dist'] = diff['x (px)']
            df['y dist'] = diff['y (px)']
            df['euclid'] = (df['x dist'].mul(df['x dist']).add(
                df['y dist'].mul(df['y dist'])))**(1 / 2)
            print(df)
            df.to_excel(
                w,
                sheet_name=f'{self.exposure_time}ms-{self.video_bubble_idx}f',
                index=False)
            print('exported')
        self.export_cnt += 1

    def export_frame(self, frame):
        folder = "lifetime_export\\reference"
        cv2.imwrite(
            f'{folder}\\{self.exposure_time}ms-{self.video_bubble_idx}f.jpg',
            frame)
