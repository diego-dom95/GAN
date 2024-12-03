import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
from pycbc.types import TimeSeries
import os
import matplotlib.pyplot as plt
import concurrent.futures
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Custom WaveformData class to hold waveform data and parameters
class WaveformData:
    def __init__(self, q=None, spin1=None, spin2=None, file_path=None):
        self.det_h1 = Detector('H1')
        self.params = {
            'q': q,
            'spin1': spin1,
            'spin2': spin2,
            'declination': 0.65,
            'right_ascension': 4.67,
            'polarization': 2.34,
            'inclination': 1.23,
            'coa_phase': 2.45,
            'delta_t': 1.0 / 4096,
            'f_lower': 40
        }
        self.norm_params = {'hp_mean': None, 'hp_std': None, 'hc_mean': None, 'hc_std': None}

        if file_path is not None:
            self.load_from_hdf5(file_path)
        elif q is not None and spin1 is not None and spin2 is not None:
            self.generate_waveform()
        else:
            raise ValueError("Provide either file_path or q, spin1, and spin2 to initialize WaveformData.")

    def generate_waveform(self):
        apx = 'SEOBNRv4'
        hp, hc = get_td_waveform(
            approximant=apx,
            mass1=20,
            mass2=20 / self.params['q'],
            spin1z=self.params['spin1'],
            spin2z=self.params['spin2'],
            inclination=self.params['inclination'],
            coa_phase=self.params['coa_phase'],
            delta_t=self.params['delta_t'],
            f_lower=self.params['f_lower']
        )
        self.hp = hp
        self.hc = hc
        self.norm_params['hp_mean'] = np.mean(hp)
        self.norm_params['hp_std'] = np.std(hp)
        self.norm_params['hc_mean'] = np.mean(hc)
        self.norm_params['hc_std'] = np.std(hc)

    def normalize_waveform(self):
        if self.hp is not None and self.hc is not None:
            self.hp = (self.hp - self.norm_params['hp_mean']) / self.norm_params['hp_std']
            self.hc = (self.hc - self.norm_params['hc_mean']) / self.norm_params['hc_std']
        else:
            raise ValueError("Waveform data not generated or loaded.")

    def denormalize_waveform(self):
        if self.hp is not None and self.hc is not None:
            self.hp = (self.hp * self.norm_params['hp_std']) + self.norm_params['hp_mean']
            self.hc = (self.hc * self.norm_params['hc_std']) + self.norm_params['hc_mean']
        else:
            raise ValueError("Waveform data not generated or loaded.")

    def project_to_detector(self):
        self.hp.start_time += 10  # Example start time
        self.hc.start_time += 10
        signal_h1 = self.det_h1.project_wave(
            self.hp, self.hc,
            self.params['right_ascension'],
            self.params['declination'],
            self.params['polarization']
        )
        return signal_h1

    def save_to_hdf5(self, save_folder):
        filename = f"waveform_q_{self.params['q']:.5f}_s1_{self.params['spin1']:.5f}_s2_{self.params['spin2']:.5f}.h5"
        file_path = os.path.join(save_folder, filename)
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('hp', data=self.hp)
            f.create_dataset('hc', data=self.hc)
            for key, value in self.params.items():
                f.attrs[key] = value
            for key, value in self.norm_params.items():
                f.attrs[key] = value

    def load_from_hdf5(self, file_path):
        with h5py.File(file_path, 'r') as f:
            hp_array = np.array(f['hp'])
            hc_array = np.array(f['hc'])
            self.hp = TimeSeries(hp_array, delta_t=self.params['delta_t'])
            self.hc = TimeSeries(hc_array, delta_t=self.params['delta_t'])
            self.params = {key: f.attrs[key] for key in f.attrs if key in self.params}
            self.norm_params = {key: f.attrs[key] for key in f.attrs if key in self.norm_params}

    def plot(self):
        signal_h1 = self.project_to_detector()
        plt.figure(figsize=(10, 4))
        plt.plot(signal_h1.sample_times, signal_h1, label='H1')
        plt.ylabel('Strain')
        plt.xlabel('Time (s)')
        #plt.xlim(9.5, 10.01)
        #plt.ylim(4e-20, -4e-20)
        plt.title(f"Gravitational Waveform q={self.params['q']}, spin s1={self.params['spin1']} and spin s2={self.params['spin2']}")
        plt.legend()
        plt.show()

# Function to generate waveforms in parallel
def generate_waveforms_in_parallel(params_list, output_dir):
    import concurrent.futures

    # Ask user for the number of cores to use
    num_cores = int(input("Enter the number of cores to use for parallel generation: "))

    # Check if output directory exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    def process_params(params):
        q, spin1, spin2 = params
        waveform = WaveformData(q=q, spin1=spin1, spin2=spin2)
        waveform.save_to_hdf5(output_dir)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
        list(tqdm(executor.map(process_params, params_list), total=len(params_list), desc="Generating waveforms"))
    
    print(f'Waveform generation completed. Files saved in {output_dir}/')