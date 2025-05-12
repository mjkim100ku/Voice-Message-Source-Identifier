import math
import binascii
from typing import Optional
from bitreader.bitreader import BitReader
from aacwindowgrouping import window_grouping
from aachuffmanutil import hcod_sf
from aachuffmanutil import hcod_2step
from aachuffmanutil import hcod_binary
from aachuffmanutil import hcod
from aachuffmanutil import t_huffman_env_bal_3_0dB, f_huffman_env_bal_3_0dB
from aachuffmanutil import t_huffman_env_bal_1_5dB, f_huffman_env_bal_1_5dB
from aachuffmanutil import t_huffman_env_3_0dB, f_huffman_env_3_0dB
from aachuffmanutil import t_huffman_env_1_5dB, f_huffman_env_1_5dB
from aachuffmanutil import t_huffman_noise_bal_3_0dB, t_huffman_noise_3_0dB
from aachuffmanutil import sbr_huff_dec
from aacsbrtables import derive_sbr_tables

MaxBitsLeft = 131072

class ADTS:
    def __init__(self, 
                 Bitrate: int = 0, 
                 ChannelConfiguration: int = 0, 
                 Layer: int = 0, 
                 MpegVersion: int = 0, 
                 Profile: int = 0, 
                 SamplingFrequency: int = 0, 
                 VbrMode: bool = False, 
                 Frame_length: int = 0, 
                 reader = None, 
                 aac_frame_length: int = 0, 
                 sfi: int = 0, 
                 num_raw_data_blocks: int = 0, 
                 protection_absent: bool = False, 
                 Single_channel_elements: list = None, 
                 Channel_pair_elements: list = None, 
                 Coupling_channel_elements: list = None, 
                 Lfe_channel_elements: list = None, 
                 Data_stream_elements: list = None, 
                 Program_config_elements: list = None, 
                 Fill_elements: list = None):
        self.Bitrate = Bitrate
        self.ChannelConfiguration = ChannelConfiguration
        self.Layer = Layer
        self.MpegVersion = MpegVersion
        self.Profile = Profile
        self.SamplingFrequency = SamplingFrequency
        self.VbrMode = VbrMode
        self.Frame_length = Frame_length

        self.reader = reader if reader is not None else BitReader(binascii.unhexlify('00'))
        self.aac_frame_length = aac_frame_length
        self.sfi = sfi
        self.num_raw_data_blocks = num_raw_data_blocks
        self.protection_absent = protection_absent

        self.Single_channel_elements = Single_channel_elements if Single_channel_elements is not None else []
        self.Channel_pair_elements = Channel_pair_elements if Channel_pair_elements is not None else []
        self.Coupling_channel_elements = Coupling_channel_elements if Coupling_channel_elements is not None else []
        self.Lfe_channel_elements = Lfe_channel_elements if Lfe_channel_elements is not None else []
        self.Data_stream_elements = Data_stream_elements if Data_stream_elements is not None else []
        self.Program_config_elements = Program_config_elements if Program_config_elements is not None else []
        self.Fill_elements = Fill_elements if Fill_elements is not None else []

    ################################################################################
    ## Table 1.A.5 – Syntax of adts_frame()
    ################################################################################    
    def adts_frame(self) -> None:
        
        if self.reader.HasByteLeft():
            err = self.adts_fixed_header()
            if err is not None:
                return err

        if self.reader.HasByteLeft():
            self.adts_variable_header()

        self.Frame_length = 1024

        if self.aac_frame_length == 0:
            return [], [], None

        if self.num_raw_data_blocks == 0:
            self.adts_error_check()
            section_lengths, huffmancodebooks, err = self.raw_data_block()
            if err is not None:
                return section_lengths, huffmancodebooks, err
        else:
            self.adts_header_error_check()
            for i in range(self.num_raw_data_blocks + 1):
                section_lengths, huffmancodebooks, err = self.raw_data_block()
                if err is not None:
                    return err
                self.adts_raw_data_block_error_check()

        return section_lengths, huffmancodebooks, None
    
    ################################################################################
    ## Table 1.A.6 – Syntax of adts_fixed_header()
    ################################################################################
    def adts_fixed_header(self) -> None:
        sync_word_count = 0
        while sync_word_count < 3 and self.reader.HasBitLeft():  # syncword 0xfff
            val, err = self.reader.ReadBitsAsUInt8(4)
            if err is not None:
                return err
            if val == 0x0f:
                sync_word_count += 1
            else:
                sync_word_count = 0

        if self.reader.HasBytesLeft(2):
            self.MpegVersion, _ = self.reader.ReadBit()
            self.Layer, _ = self.reader.ReadBitsAsUInt8(2)
            if self.Layer != 0:
                return ValueError(f"ADTS Layer ({self.Layer}) must be 0")

            self.protection_absent, _ = self.reader.ReadBitAsBool()
            self.Profile, _ = self.reader.ReadBitsAsUInt8(2)
            self.Profile += 1

            self.sfi, _ = self.reader.ReadBitsAsUInt8(4)
            if self.sfi > 12:
                return ValueError(f"Sampling Frequency Index ({self.sfi}) out of acceptable range (0-12)")

            self.SamplingFrequency = SamplingFrequency[self.sfi]
            self.reader.SkipBits(1)
            self.ChannelConfiguration, _ = self.reader.ReadBitsAsUInt8(3)
            self.reader.SkipBits(1)
            self.reader.SkipBits(1)

        return None
    
    ################################################################################
    ## Table 1.A.7 – Syntax of adts_variable_header()
    ################################################################################
    def adts_variable_header(self) -> None:
        if self.reader.HasBytesLeft(4):
            self.reader.SkipBits(1)
            self.reader.SkipBits(1)
            self.aac_frame_length, _ = self.reader.ReadBitsAsUInt16(13)
            adts_buffer_fullness, _ = self.reader.ReadBitsAsUInt16(11)
            self.num_raw_data_blocks, _ = self.reader.ReadBitsAsUInt8(2)

            if adts_buffer_fullness == 0x7ff:
                self.VbrMode = True
            else:
                self.VbrMode = False

            self.Bitrate = self.SamplingFrequency // 1024
            self.Bitrate *= self.aac_frame_length * 8

    ################################################################################
    ## Table 1.A.8 – Syntax of adts_error_check
    ################################################################################
    def adts_error_check(self) -> None:
        if not self.protection_absent:
            self.reader.SkipBits(16)

    ################################################################################
    ## Table 1.A.9 – Syntax of adts_header_error_check
    ################################################################################
    def adts_header_error_check(self) -> None:
        data = adts_header_error_check()

        if not self.protection_absent:
            data.Raw_data_block_position = [0] * (self.num_raw_data_blocks + 1)
            for i in range(1, self.num_raw_data_blocks + 1):
                data.Raw_data_block_position[i], _ = self.reader.ReadBitsAsUInt16(16)
            data.Crc_check, _ = self.reader.ReadBitsAsUInt16(16)

    ################################################################################
    ## Table 1.A.10 – Syntax of adts_raw_data_block_error_check()
    ################################################################################
    def adts_raw_data_block_error_check(self) -> None:
        if not self.protection_absent:
            self.reader.SkipBits(16)

    ################################################################################
    ## Table 4.2 – Syntax of program_config_element()
    ################################################################################
    def program_config_element(self) -> 'program_config_element':
        e = program_config_element()
        e.Element_instance_tag, _ = self.reader.ReadBitsAsUInt8(4)

        e.Object_type, _ = self.reader.ReadBitsAsUInt8(2)
        e.Sampling_frequency_index, _ = self.reader.ReadBitsAsUInt8(4)
        e.Num_front_channel_elements, _ = self.reader.ReadBitsAsUInt8(4)
        e.Num_side_channel_elements, _ = self.reader.ReadBitsAsUInt8(4)
        e.Num_back_channel_elements, _ = self.reader.ReadBitsAsUInt8(4)
        e.Num_lfe_channel_elements, _ = self.reader.ReadBitsAsUInt8(2)
        e.Num_assoc_data_elements, _ = self.reader.ReadBitsAsUInt8(3)
        e.Num_valid_cc_elements, _ = self.reader.ReadBitsAsUInt8(4)

        e.Mono_mixdown_present, _ = self.reader.ReadBitAsBool()
        if e.Mono_mixdown_present:
            e.Mono_mixdown_element_num, _ = self.reader.ReadBitsAsUInt8(4)

        e.Stereo_mixdown_present, _ = self.reader.ReadBitAsBool()
        if e.Stereo_mixdown_present:
            e.Stereo_mixdown_element_num, _ = self.reader.ReadBitsAsUInt8(4)

        e.Matrix_mixdown_idx_present, _ = self.reader.ReadBitAsBool()
        if e.Matrix_mixdown_idx_present:
            e.Matrix_mixdown_idx, _ = self.reader.ReadBitsAsUInt8(2)
            e.Pseudo_surround_enable, _ = self.reader.ReadBitAsBool()

        e.Front_element_is_cpe = [False] * e.Num_front_channel_elements
        e.Front_element_tag_select = [0] * e.Num_front_channel_elements
        for i in range(e.Num_front_channel_elements):
            e.Front_element_is_cpe[i], _ = self.reader.ReadBitAsBool()
            e.Front_element_tag_select[i], _ = self.reader.ReadBitsAsUInt8(4)

        e.Side_element_is_cpe = [False] * e.Num_side_channel_elements
        e.Side_element_tag_select = [0] * e.Num_side_channel_elements
        for i in range(e.Num_side_channel_elements):
            e.Side_element_is_cpe[i], _ = self.reader.ReadBitAsBool()
            e.Side_element_tag_select[i], _ = self.reader.ReadBitsAsUInt8(4)

        e.Back_element_is_cpe = [False] * e.Num_back_channel_elements
        e.Back_element_tag_select = [0] * e.Num_back_channel_elements
        for i in range(e.Num_back_channel_elements):
            e.Back_element_is_cpe[i], _ = self.reader.ReadBitAsBool()
            e.Back_element_tag_select[i], _ = self.reader.ReadBitsAsUInt8(4)

        e.Lfe_element_tag_select = [0] * e.Num_lfe_channel_elements
        for i in range(e.Num_lfe_channel_elements):
            e.Lfe_element_tag_select[i], _ = self.reader.ReadBitsAsUInt8(4)

        e.Assoc_data_element_tag_select = [0] * e.Num_assoc_data_elements
        for i in range(e.Num_assoc_data_elements):
            e.Assoc_data_element_tag_select[i], _ = self.reader.ReadBitsAsUInt8(4)

        e.Cc_element_is_ind_sw = [False] * e.Num_valid_cc_elements
        e.Valid_cc_element_tag_select = [0] * e.Num_valid_cc_elements
        for i in range(e.Num_valid_cc_elements):
            e.Cc_element_is_ind_sw[i], _ = self.reader.ReadBitAsBool()
            e.Valid_cc_element_tag_select[i], _ = self.reader.ReadBitsAsUInt8(4)

        self.reader.ByteAlign()
        e.Comment_field_bytes, _ = self.reader.ReadBitsAsUInt8(8)
        e.Comment_field_data, _ = self.reader.ReadBitsToByteArray(e.Comment_field_bytes * 8)

        return e


    ################################################################################
    ## Table 4.3 – Syntax of top level payload for audio object types AAC Main,
    ##             SSR, LC, and LTP (raw_data_block())
    ################################################################################
    def raw_data_block(self) -> None:
        id_syn_ele = 0
        id_syn_ele_Previous = 0
        err = None
        huffmancodebooks = []
        section_lengths = []

        while id_syn_ele != ID_END:
            id_syn_ele_Previous = id_syn_ele
            id_syn_ele, _ = self.reader.ReadBits(3)

            if id_syn_ele == ID_SCE:
                e, section_length, huffmancodebook, err = self.single_channel_element()
                self.Single_channel_elements.append(e)
                huffmancodebooks.append(huffmancodebook)
                section_lengths.append(section_length)
            elif id_syn_ele == ID_CPE:
                e, section_length1, section_length2, huffmancodebook1, huffmancodebook2, err = self.channel_pair_element()
                self.Channel_pair_elements.append(e)
                huffmancodebooks.append(huffmancodebook1)
                huffmancodebooks.append(huffmancodebook2)
                section_lengths.append(section_length1)
                section_lengths.append(section_length2)
            elif id_syn_ele == ID_CCE:
                e, err = self.coupling_channel_element()
                self.Coupling_channel_elements.append(e)
            elif id_syn_ele == ID_LFE:
                e, err = self.lfe_channel_element()
                self.Lfe_channel_elements.append(e)
            elif id_syn_ele == ID_DSE:
                e = self.data_stream_element()
                self.Data_stream_elements.append(e)
            elif id_syn_ele == ID_PCE:
                e = self.program_config_element()
                self.Program_config_elements.append(e)
            elif id_syn_ele == ID_FIL:
                e, err = self.fill_element(id_syn_ele_Previous)
                self.Fill_elements.append(e)
            elif id_syn_ele == ID_END:
                break
            else:
                err = ValueError(f"Error: Unsupported id_syn_ele: {id_syn_ele}")

            if id_syn_ele != ID_END and not self.reader.HasBitLeft():
                err = ValueError(f"Error: Buffer empty parsing id_syn_ele {id_syn_ele}")

            if err is not None:
                return section_lengths, huffmancodebooks, err

        self.reader.ByteAlign()
        return section_lengths, huffmancodebooks, None
    
    ################################################################################
    ## Table 4.4 – Syntax of single_channel_element()
    ################################################################################
    def single_channel_element(self) -> tuple:
        e = single_channel_element()
        e.Element_instance_tag, _ = self.reader.ReadBitsAsUInt8(4)
        e.Channel_stream, section_length, huffmancodebook, err = self.individual_channel_stream(False, False, None)
        return e, section_length, huffmancodebook, err

    ################################################################################
    ## Table 4.5 – Syntax of channel_pair_element()
    ################################################################################
    def channel_pair_element(self) -> tuple:
        e = channel_pair_element()
        e.Element_instance_tag, _ = self.reader.ReadBitsAsUInt8(4)

        e.Common_window, _ = self.reader.ReadBitAsBool()
        if e.Common_window:
            e.Ics_info, err = self.ics_info(e.Common_window)
            if err is not None:
                return e, err

            ms_mask_present, _ = self.reader.ReadBitsAsUInt8(2)
            if ms_mask_present == 3:
                return e, ValueError(f"Error: ms_mask_present ({ms_mask_present}) out of range")

            if ms_mask_present == 1:
                e.Ms_used = [[False] * e.Ics_info.Max_sfb for _ in range(e.Ics_info.num_window_groups)]
                for g in range(e.Ics_info.num_window_groups):
                    for sfb in range(e.Ics_info.Max_sfb):
                        e.Ms_used[g][sfb], _ = self.reader.ReadBitAsBool()

        e.Channel_stream1, section_length1, huffmancodebook1, err = self.individual_channel_stream(e.Common_window, False, e.Ics_info)
        if err is not None:
            return e, huffmancodebook1, err


        e.Channel_stream2, section_length2, huffmancodebook2, err = self.individual_channel_stream(e.Common_window, False, e.Ics_info)
        return e, section_length1, section_length2, huffmancodebook1, huffmancodebook2, err

    ################################################################################
    ## Table 4.6 – Syntax of ics_info()
    ################################################################################
    def ics_info(self, common_window: bool) -> tuple:
        info = ics_info()
        if self.reader.ReadBitsAsUInt8(1)[0] != 0:
            err = ValueError("Error: ics_reserved_bit must equal 0")
            return None, err
    
        info.Window_sequence, _ = self.reader.ReadBits(2)
        info.Window_shape, _ = self.reader.ReadBitsAsUInt8(1)
    
        if info.Window_sequence == EIGHT_SHORT_SEQUENCE:
            info.Max_sfb, _ = self.reader.ReadBits(4)
            info.Scale_factor_grouping, _ = self.reader.ReadBits(7)
    
            window_grouping(info, self.sfi, self.Frame_length)
            if info.Max_sfb > info.num_swb:
                err = ValueError(f"Error: ics_info.Max_sfb ({info.Max_sfb}) must be less than ics_info.num_swb ({info.num_swb})")
                return None, err
        else:
            window_grouping(info, self.sfi, self.Frame_length)
    
            info.Max_sfb, _ = self.reader.ReadBits(6)
            if info.Max_sfb > info.num_swb:
                err = ValueError(f"Error: ics_info.Max_sfb ({info.Max_sfb}) must be less than ics_info.num_swb ({info.num_swb})")
                return None, err
    
            info.Predictor_data_present, _ = self.reader.ReadBitAsBool()
            if info.Predictor_data_present:
                if self.Profile == AUDIO_OBJECT_TYPE_AAC_MAIN:
                    info.Predictor_reset, _ = self.reader.ReadBitAsBool()
                    if info.Predictor_reset:
                        info.Predictor_reset_group_num, _ = self.reader.ReadBits(5)
    
                    PRED_SFB_MAX = min(int(info.Max_sfb), int(Aac_PRED_SFB_MAX[self.sfi]))
                    info.Prediction_used = [False] * PRED_SFB_MAX
                    for sfb in range(PRED_SFB_MAX):
                        info.Prediction_used[sfb], _ = self.reader.ReadBitAsBool()
    
                else:
                    info.Ltp_data_present, _ = self.reader.ReadBitAsBool()
                    if info.Ltp_data_present:
                        info.Ltp_data, err = self.ltp_data(info)
                    if common_window:
                        info.Ltp_data_present, _ = self.reader.ReadBitAsBool()
                        if info.Ltp_data_present:
                            info.Ltp_data, err = self.ltp_data(info)
    
        return info, None

    
    ################################################################################
    ## Table 4.7 – Syntax of pulse_data()
    ################################################################################
    def pulse_data(self) -> 'pulse_data':
        data = pulse_data()
        data.Number_pulse, _ = self.reader.ReadBitsAsUInt8(2)
        data.Pulse_start_sfb, _ = self.reader.ReadBitsAsUInt8(6)
        data.Pulse_amp = [0] * data.Number_pulse
        data.Pulse_offset = [0] * data.Number_pulse

        for i in range(data.Number_pulse):
            data.Pulse_offset[i], _ = self.reader.ReadBitsAsUInt8(5)
            data.Pulse_amp[i], _ = self.reader.ReadBitsAsUInt8(4)

        return data

    ################################################################################
    ## Table 4.8 – Syntax of coupling_channel_element()
    ################################################################################
    def coupling_channel_element(self) -> tuple:
        e = coupling_channel_element()
        e.Element_instance_tag, _ = self.reader.ReadBitsAsUInt8(4)

        e.Ind_sw_cce_flag, _ = self.reader.ReadBitAsBool()
        e.Num_coupled_elements, _ = self.reader.ReadBitsAsUInt8(3)
        num_gain_element_lists = 0

        e.Cc_target_is_cpe = [False] * e.Num_coupled_elements
        e.Cc_target_tag_select = [0] * e.Num_coupled_elements
        for c in range(e.Num_coupled_elements):
            num_gain_element_lists += 1
            e.Cc_target_is_cpe[c], _ = self.reader.ReadBitAsBool()
            e.Cc_target_tag_select[c], _ = self.reader.ReadBitsAsUInt8(4)
            if e.Cc_target_is_cpe[c]:
                if e.Cc_l is None and e.Cc_r is None:
                    e.Cc_l = [False] * e.Num_coupled_elements
                    e.Cc_r = [False] * e.Num_coupled_elements
                e.Cc_l[c], _ = self.reader.ReadBitAsBool()
                e.Cc_r[c], _ = self.reader.ReadBitAsBool()
                if e.Cc_l[c] and e.Cc_r[c]:
                    num_gain_element_lists += 1

        e.Cc_domain, _ = self.reader.ReadBitAsBool()
        e.Gain_element_sign, _ = self.reader.ReadBitAsBool()
        e.Gain_element_scale, _ = self.reader.ReadBitsAsUInt8(2)

        e.Channel_stream, err = self.individual_channel_stream(False, False, None)
        if err is not None:
            return e, err

        e.Common_gain_element_present = [False] * num_gain_element_lists
        e.Common_gain_element = [0] * num_gain_element_lists
        e.dpcm_gain_element = [[[] for _ in range(self.sfi)] for _ in range(num_gain_element_lists)]
        for c in range(1, num_gain_element_lists):
            cge = False
            if e.Ind_sw_cce_flag:
                cge = True
            else:
                e.Common_gain_element_present[c], _ = self.reader.ReadBitAsBool()
                cge = e.Common_gain_element_present[c]

            if cge:
                e.Common_gain_element[c], err = hcod_sf(self.reader)
            else:
                info = e.Channel_stream.Ics_info
                e.dpcm_gain_element[c] = [[0] * info.Max_sfb for _ in range(info.num_window_groups)]
                for g in range(info.num_window_groups):
                    for sfb in range(info.Max_sfb):
                        if info.sfb_cb[g][sfb] != ZERO_HCB:
                            e.dpcm_gain_element[c][g][sfb], err = hcod_sf(self.reader)
                            if err is not None:
                                return e, err

        return e, err

    ################################################################################
    ## Table 4.9 – Syntax of lfe_channel_element()
    ################################################################################
    def lfe_channel_element(self) -> tuple:
        e = lfe_channel_element()
        e.Element_instance_tag, _ = self.reader.ReadBitsAsUInt8(4)

        e.Channel_stream, err = self.individual_channel_stream(False, False, None)
        return e, err

    ################################################################################
    ## Table 4.10 – Syntax of data_stream_element()
    ################################################################################
    def data_stream_element(self) -> 'data_stream_element':
        e = data_stream_element()
        e.Element_instance_tag, _ = self.reader.ReadBitsAsUInt8(4)

        e.Data_byte_align_flag, _ = self.reader.ReadBitAsBool()
        e.Count, _ = self.reader.ReadBitsAsUInt8(8)
        if e.Count == 255:
            e.Esc_count, _ = self.reader.ReadBitsAsUInt8(8)
            e.Count += e.Esc_count

        if e.Data_byte_align_flag:
            self.reader.ByteAlign()

        e.Data_stream_byte = [[0] * e.Count for _ in range(e.Element_instance_tag + 1)]
        for i in range(e.Count):
            e.Data_stream_byte[e.Element_instance_tag][i], _ = self.reader.ReadBitsAsUInt8(8)

        return e


    ################################################################################
    ## Table 4.11 – Syntax of fill_element()
    ################################################################################
    def fill_element(self, id_syn_ele: int) -> tuple:
        e = fill_element()
        e.Count, _ = self.reader.ReadBitsAsUInt16(4)

        if e.Count == 15:
            e.Esc_count, _ = self.reader.ReadBitsAsUInt8(8)
            e.Count += e.Esc_count - 1

        cnt = int(e.Count)
        while cnt > 0:
            sub, e.Extension_payload, err = self.extension_payload(cnt, id_syn_ele)
            cnt -= sub
            if err is not None:
                return e, err

        return e, None

    ################################################################################
    ## Table 4.12 – Syntax of gain_control_data()
    ################################################################################
    def gain_control_data(self, info: 'ics_info') -> 'gain_control_data':
        data = gain_control_data()

        data.Max_band, _ = self.reader.ReadBitsAsUInt8(2)
        data.Adjust_num = [[0] * 1 for _ in range(data.Max_band)]
        data.Alevcode = [[[0] * 1 for _ in range(1)] for _ in range(data.Max_band)]
        data.Aloccode = [[[0] * 1 for _ in range(1)] for _ in range(data.Max_band)]

        if info.Window_sequence == ONLY_LONG_SEQUENCE:
            for bd in range(1, data.Max_band):
                data.Adjust_num[bd] = [0] * 1
                data.Alevcode[bd] = [[0] * 0 for _ in range(1)]
                data.Aloccode[bd] = [[0] * 0 for _ in range(1)]
                for wd in range(len(data.Adjust_num[bd])):
                    data.Adjust_num[bd][wd], _ = self.reader.ReadBitsAsUInt8(3)
                    data.Alevcode[bd][wd] = [0] * data.Adjust_num[bd][wd]
                    data.Aloccode[bd][wd] = [0] * data.Adjust_num[bd][wd]
                    for ad in range(data.Adjust_num[bd][wd]):
                        data.Alevcode[bd][wd][ad], _ = self.reader.ReadBitsAsUInt8(4)
                        data.Aloccode[bd][wd][ad], _ = self.reader.ReadBitsAsUInt8(5)

        elif info.Window_sequence == LONG_START_SEQUENCE:
            for bd in range(1, data.Max_band):
                data.Adjust_num[bd] = [0] * 2
                data.Alevcode[bd] = [[0] * 0 for _ in range(2)]
                data.Aloccode[bd] = [[0] * 0 for _ in range(2)]
                for wd in range(len(data.Adjust_num[bd])):
                    data.Adjust_num[bd][wd], _ = self.reader.ReadBitsAsUInt8(3)
                    data.Alevcode[bd][wd] = [0] * data.Adjust_num[bd][wd]
                    data.Aloccode[bd][wd] = [0] * data.Adjust_num[bd][wd]
                    for ad in range(data.Adjust_num[bd][wd]):
                        data.Alevcode[bd][wd][ad], _ = self.reader.ReadBitsAsUInt8(4)
                        if wd == 0:
                            data.Aloccode[bd][wd][ad], _ = self.reader.ReadBitsAsUInt8(4)
                        else:
                            data.Aloccode[bd][wd][ad], _ = self.reader.ReadBitsAsUInt8(2)

        elif info.Window_sequence == EIGHT_SHORT_SEQUENCE:
            for bd in range(1, data.Max_band):
                data.Adjust_num[bd] = [0] * 8
                data.Alevcode[bd] = [[0] * 0 for _ in range(8)]
                data.Aloccode[bd] = [[0] * 0 for _ in range(8)]
                for wd in range(len(data.Adjust_num[bd])):
                    data.Adjust_num[bd][wd], _ = self.reader.ReadBitsAsUInt8(3)
                    data.Alevcode[bd][wd] = [0] * data.Adjust_num[bd][wd]
                    data.Aloccode[bd][wd] = [0] * data.Adjust_num[bd][wd]
                    for ad in range(data.Adjust_num[bd][wd]):
                        data.Alevcode[bd][wd][ad], _ = self.reader.ReadBitsAsUInt8(4)
                        data.Aloccode[bd][wd][ad], _ = self.reader.ReadBitsAsUInt8(2)

        elif info.Window_sequence == LONG_STOP_SEQUENCE:
            for bd in range(1, data.Max_band):
                data.Adjust_num[bd] = [0] * 2
                data.Alevcode[bd] = [[0] * 0 for _ in range(2)]
                data.Aloccode[bd] = [[0] * 0 for _ in range(2)]
                for wd in range(len(data.Adjust_num[bd])):
                    data.Adjust_num[bd][wd], _ = self.reader.ReadBitsAsUInt8(3)
                    data.Alevcode[bd][wd] = [0] * data.Adjust_num[bd][wd]
                    data.Aloccode[bd][wd] = [0] * data.Adjust_num[bd][wd]
                    for ad in range(data.Adjust_num[bd][wd]):
                        data.Alevcode[bd][wd][ad], _ = self.reader.ReadBitsAsUInt8(4)
                        if wd == 0:
                            data.Aloccode[bd][wd][ad], _ = self.reader.ReadBitsAsUInt8(4)
                        else:
                            data.Aloccode[bd][wd][ad], _ = self.reader.ReadBitsAsUInt8(5)
        else:
            return None

        return data

    ################################################################################
    ## Table 4.50 – Syntax of individual_channel_stream()
    ################################################################################
    def individual_channel_stream(self, common_window: bool, scale_flag: bool, info: 'ics_info') -> tuple:
        s = individual_channel_stream()
        s.Global_gain, _ = self.reader.ReadBitsAsUInt8(8)

        if not common_window and not scale_flag:
            s.Ics_info, err = self.ics_info(common_window)
            if err is not None:
                return s, err
        else:
            s.Ics_info = info

        s.Section_data, section_length, err = self.section_data(s.Ics_info)
        if err is not None:
            return s, err

        s.Scale_factor_data, err = self.scale_factor_data(s.Ics_info)
        if err is not None:
            return s, err

        if not scale_flag:
            s.Pulse_data_present, _ = self.reader.ReadBitAsBool()
            if s.Pulse_data_present:
                s.Pulse_data = self.pulse_data()

            s.Tns_data_present, _ = self.reader.ReadBitAsBool()
            if s.Tns_data_present:
                s.Tns_data = self.tns_data(s.Ics_info)

            s.Gain_control_data_present, _ = self.reader.ReadBitAsBool()
            if s.Gain_control_data_present:
                self.gain_control_data(s.Ics_info)

        s.Spectral_data, huffmancodebook, err = self.spectral_data(s.Ics_info, s.Section_data)
        # else:
        #     self.reader.SkipBits(14)  # length_of_reordered_spectral_data
        #     self.reader.SkipBits(6)   # length_of_longest_codeword
        #     self.reordered_spectral_data()

        return s, section_length, huffmancodebook, err

    ################################################################################
    ## Table 4.51 – Syntax of reordered_spectral_data ()
    ################################################################################
    # def reordered_spectral_data(self):
    
    ################################################################################
    ## Table 4.52 – Syntax of section_data()
    ################################################################################
    def section_data(self, info: 'ics_info') -> tuple:
        data = section_data()
        section_length = []
        aacSectionDataResilienceFlag = False

        if info.Window_sequence == EIGHT_SHORT_SEQUENCE:
            bits = 3
        else:
            bits = 5

        sect_esc_val = (1 << bits) - 1
        data.Sect_cb = [[] for _ in range(info.num_window_groups)]
        data.sect_start = [[] for _ in range(info.num_window_groups)]
        data.sect_end = [[] for _ in range(info.num_window_groups)]
        data.num_sec = [0] * info.num_window_groups
        info.sfb_cb = [[] for _ in range(info.num_window_groups)]
        section_length = [[] for _ in range(info.num_window_groups)]

        for g in range(len(data.Sect_cb)):
            i = 0
            k = 0
            while k < info.Max_sfb:
                if aacSectionDataResilienceFlag:
                    val, _ = self.reader.ReadBits(5)
                    data.Sect_cb[g].append(val)
                else:
                    val, _ = self.reader.ReadBits(4)
                    data.Sect_cb[g].append(val)

                sect_len = 0
                sect_len_incr = 0

                if not aacSectionDataResilienceFlag or data.Sect_cb[g][i] < 11 or (data.Sect_cb[g][i] > 11 and data.Sect_cb[g][i] < 16):
                    sect_len_incr, _ = self.reader.ReadBitsAsUInt8(bits)
                    while sect_len_incr == sect_esc_val:
                        sect_len += sect_len_incr
                        sect_len_incr, err = self.reader.ReadBitsAsUInt8(bits)
                        if err is not None:
                            return data, err
                else:
                    sect_len_incr = 1

                sect_len += sect_len_incr
                section_length[g].append(sect_len)
                data.sect_start[g].append(k)
                data.sect_end[g].append(k + sect_len)

                for j in range(sect_len):
                    info.sfb_cb[g].append(data.Sect_cb[g][i])

                k += sect_len
                i += 1

                upper_bound = info.Max_sfb
                if info.Window_sequence == EIGHT_SHORT_SEQUENCE:
                    upper_bound = 8 * 15
                if k > upper_bound or i > upper_bound:
                    err = f"Error: Section Codebook param out of bounds({upper_bound}). End ({k}), Index ({i})"
                    return data, err

            data.num_sec[g] = i
            if k != info.Max_sfb:
                err = f"Error: Total length ({k}) does not equal Max_sfb ({info.Max_sfb})"
                return data, err

        return data, section_length, None

    ################################################################################
    ## Table 4.53 – Syntax of scale_factor_data()
    ################################################################################
    def scale_factor_data(self, info: 'ics_info') -> tuple:
        data = scale_factor_data()
        noise_pcm_flag = True

        data.dpcm_is_position = [[0] * info.Max_sfb for _ in range(info.num_window_groups)]
        data.dpcm_noise_nrg = [[0] * info.Max_sfb for _ in range(info.num_window_groups)]
        data.dpcm_sf = [[0] * info.Max_sfb for _ in range(info.num_window_groups)]

        for g in range(info.num_window_groups):
            for sfb in range(info.Max_sfb):
                if info.sfb_cb[g][sfb] != ZERO_HCB:
                    if is_intensity(info, g, sfb) != 0:
                        data.dpcm_is_position[g][sfb], err = hcod_sf(self.reader)
                    else:
                        if is_noise(info, g, sfb):
                            if noise_pcm_flag:
                                noise_pcm_flag = False
                                data.dpcm_noise_nrg[g][sfb], _ = self.reader.ReadBitsAsUInt16(9)
                            else:
                                val, err = hcod_sf(self.reader)
                                data.dpcm_noise_nrg[g][sfb] = val
                        else:
                            data.dpcm_sf[g][sfb], err = hcod_sf(self.reader)
                        if err is not None:
                            return data, err
                else:
                    data.dpcm_sf[g][sfb] = 0

        return data, None

    ################################################################################
    ## Table 4.54 – Syntax of tns_data()
    ################################################################################
    def tns_data(self, info: 'ics_info') -> 'tns_data':
        data = tns_data()

        filt_bits = 2
        len_bits = 6
        order_bits = 5
        if info.Window_sequence == EIGHT_SHORT_SEQUENCE:
            filt_bits = 1
            len_bits = 4
            order_bits = 3

        data.N_filt = [0] * info.num_windows
        data.Len = [[] for _ in range(info.num_windows)]
        data.Order = [[] for _ in range(info.num_windows)]
        data.Direction = [[] for _ in range(info.num_windows)]
        data.Coef_res = [0] * info.num_windows
        data.Coef_compress = [[] for _ in range(info.num_windows)]
        data.Coef = [[] for _ in range(info.num_windows)]

        for w in range(info.num_windows):
            data.N_filt[w], _ = self.reader.ReadBitsAsUInt8(filt_bits)
            if data.N_filt[w] != 0:
                data.Coef_res[w], _ = self.reader.ReadBitsAsUInt8(1)

            data.Len[w] = [0] * data.N_filt[w]
            data.Order[w] = [0] * data.N_filt[w]
            data.Direction[w] = [False] * data.N_filt[w]
            data.Coef_compress[w] = [0] * data.N_filt[w]
            data.Coef[w] = [[] for _ in range(data.N_filt[w])]

            for filt in range(data.N_filt[w]):
                data.Len[w][filt], _ = self.reader.ReadBitsAsUInt8(len_bits)
                data.Order[w][filt], _ = self.reader.ReadBitsAsUInt8(order_bits)
                if data.Order[w][filt] != 0:
                    data.Direction[w][filt], _ = self.reader.ReadBitAsBool()
                    data.Coef_compress[w][filt], _ = self.reader.ReadBitsAsUInt8(1)

                    coef_bits = data.Coef_res[w] + 3 - data.Coef_compress[w][filt]
                    data.Coef[w][filt] = [0] * data.Order[w][filt]

                    for i in range(data.Order[w][filt]):
                        data.Coef[w][filt][i], _ = self.reader.ReadBitsAsUInt8(coef_bits)

        return data

    ################################################################################
    ## Table 4.55 – Syntax of ltp_data()
    ################################################################################
    def ltp_data(self, info: 'ics_info') -> tuple:
        data = ltp_data()
        if self.Profile == AUDIO_OBJECT_TYPE_ER_AAC_LD:
            ltp_lag_update, _ = self.reader.ReadBitAsBool()
            if ltp_lag_update:
                data.Ltp_lag, _ = self.reader.ReadBitsAsUInt(10)
            else:
                data.Ltp_lag = info.Ltp_data.Ltp_lag

            if data.Ltp_lag > (self.Frame_length << 1):
                return data, f"Error: Ltp_lag ({data.Ltp_lag}) out of range ({self.Frame_length << 1})"

            data.Ltp_coef, _ = self.reader.ReadBitsAsUInt8(3)
            data.Ltp_long_used = [False] * min(info.Max_sfb, MAX_LTP_LONG_SFB)
            for sfb in range(len(data.Ltp_long_used)):
                data.Ltp_long_used[sfb], _ = self.reader.ReadBitAsBool()

        else:
            data.Ltp_lag, _ = self.reader.ReadBitsAsUInt(11)
            if data.Ltp_lag > (self.Frame_length << 1):
                return data, f"Error: Ltp_lag ({data.Ltp_lag}) out of range ({self.Frame_length << 1})"

            data.Ltp_coef, _ = self.reader.ReadBitsAsUInt8(3)
            if info.Window_sequence != EIGHT_SHORT_SEQUENCE:
                data.Ltp_long_used = [False] * min(info.Max_sfb, MAX_LTP_LONG_SFB)
                for sfb in range(len(data.Ltp_long_used)):
                    data.Ltp_long_used[sfb], _ = self.reader.ReadBitAsBool()

        return data, None

    ################################################################################
    ## Table 4.56 – Syntax of spectral_data()
    ################################################################################
    def spectral_data(self, info: 'ics_info', sec_data: 'section_data') -> tuple:
        huffmancodebook = []
        data = spectral_data()

        for g in range(info.num_window_groups):
            for i in range(sec_data.num_sec[g]):
                if sec_data.Sect_cb[g][i] in [ZERO_HCB, NOISE_HCB, INTENSITY_HCB, INTENSITY_HCB2]:
                    continue
                else:
                    inc = 4
                    if sec_data.Sect_cb[g][i] >= FIRST_PAIR_HCB:
                        inc = 2

                    start = info.sect_sfb_offset[g][sec_data.sect_start[g][i]]
                    end = info.sect_sfb_offset[g][sec_data.sect_end[g][i]]
                    for k in range(start, end, inc):
                        if sec_data.Sect_cb[g][i] != 0:
                            if not self.reader.HasBitLeft():
                                return data, "Error: Spectral Data parsing ran out of bits"

                            val, err = hcod(self.reader, sec_data.Sect_cb[g][i])
                            if err is not None:
                                return data, err

                            data.Hcod.append(val)
                            huffmancodebook.append(sec_data.Sect_cb[g][i])

        return data, huffmancodebook, None

    ################################################################################
    ## Table 4.57 – Syntax of extension_payload()
    ################################################################################
    def extension_payload(self, cnt: int, id_adts: int) -> tuple:
        data = extension_payload()
        data.Extension_type, _ = self.reader.ReadBitsAsUInt8(4)

        if data.Extension_type == EXT_DYNAMIC_RANGE:
            cnt, data.Dynamic_range_info = self.dynamic_range_info()
            return cnt, data, None

        elif data.Extension_type == EXT_SAC_DATA:
            cnt, data.Sac_extension_data = self.sac_extension_data(cnt)
            return cnt, data, None

        elif data.Extension_type == EXT_SBR_DATA:
            cnt, data.Sbr_extension_data, err = self.sbr_extension_data(cnt, id_adts, False)
            return cnt, data, err

        elif data.Extension_type == EXT_SBR_DATA_CRC:
            cnt, data.Sbr_extension_data, err = self.sbr_extension_data(cnt, id_adts, True)
            return cnt, data, err

        elif data.Extension_type == EXT_FILL_DATA:
            data.Fill_nibble, _ = self.reader.ReadBitsAsUInt8(4)
            if int(self.reader.BitsLeft()) > cnt:
                data.Fill_byte, _ = self.reader.ReadBitsToByteArray(8 * (cnt - 1))

        elif data.Extension_type == EXT_DATA_ELEMENT:
            data.Data_element_version, _ = self.reader.ReadBitsAsUInt8(4)
            if data.Data_element_version == ANC_DATA:
                dataElementLength = 0
                while True:
                    data.DataElementLengthPart, _ = self.reader.ReadBitsAsUInt8(8)
                    dataElementLength += data.DataElementLengthPart
                    if data.DataElementLengthPart != 255:
                        break
                data.Data_element_byte, _ = self.reader.ReadBytes(dataElementLength)

        elif data.Extension_type == EXT_FILL:
            pass

        else:
            self.reader.SkipBits(8 * (cnt - 1) + 4)

        return cnt, data, None

    ################################################################################
    ## Table 4.58 – Syntax of dynamic_range_info()
    ################################################################################
    def dynamic_range_info(self) -> tuple:
        info = dynamic_range_info()

        n = 1
        drc_num_bands = 1
        info.Pce_tag_present, _ = self.reader.ReadBitAsBool()
        if info.Pce_tag_present:
            info.Pce_instance_tag, _ = self.reader.ReadBitsAsUInt8(4)
            info.Drc_tag_reserve_bits, _ = self.reader.ReadBitsAsUInt8(4)

        info.Excluded_chns_present, _ = self.reader.ReadBitAsBool()
        if info.Excluded_chns_present:
            exclude, info.Excluded_chns = self.excluded_channels()
            n += exclude

        info.Drc_bands_present, _ = self.reader.ReadBitAsBool()
        if info.Drc_bands_present:
            info.Drc_band_incr, _ = self.reader.ReadBitsAsUInt8(4)
            info.Drc_interpolation_scheme, _ = self.reader.ReadBitsAsUInt8(4)

            n += 1
            drc_num_bands += info.Drc_band_incr
            info.Drc_band_top, _ = self.reader.ReadBytes(drc_num_bands)

        info.Prog_ref_level_present, _ = self.reader.ReadBitAsBool()
        if info.Prog_ref_level_present:
            info.Prog_ref_level, _ = self.reader.ReadBitsAsUInt8(7)
            info.Prog_ref_level_reserved_bits, _ = self.reader.ReadBit()
            n += 1

        info.Dyn_range_sign = [0] * drc_num_bands
        info.Dyn_range_cnt = [0] * drc_num_bands
        for i in range(drc_num_bands):
            info.Dyn_range_sign[i], _ = self.reader.ReadBitsAsUInt8(1)
            info.Dyn_range_cnt[i], _ = self.reader.ReadBitsAsUInt8(7)
            n += 1

        return n, info

    ################################################################################
    ## Table 4.59 – Syntax of excluded_channels()
    ################################################################################
    def excluded_channels(self) -> tuple:
        data = excluded_channels()
        n = 0
        num_excl_chan = 7
        data.Exclude_mask = [False] * 7
        for i in range(len(data.Exclude_mask)):
            data.Exclude_mask[i], _ = self.reader.ReadBitAsBool()

        n += 1

        data.Additional_excluded_chns = []
        additional_excluded_chn, _ = self.reader.ReadBitAsBool()
        data.Additional_excluded_chns.append(additional_excluded_chn)
        while data.Additional_excluded_chns[n - 1]:
            for i in range(num_excl_chan, num_excl_chan + 7):
                mask, _ = self.reader.ReadBitAsBool()
                data.Exclude_mask.append(mask)

            n += 1
            num_excl_chan += 7

            additional_excluded_chn, _ = self.reader.ReadBitAsBool()
            data.Additional_excluded_chns.append(additional_excluded_chn)

        return n, data

    ################################################################################
    ## Table 4.60 – Syntax of ms_data()
    ################################################################################
    # def ms_data(self):
    #     for g in range(num_window_groups):
    #         for sfb in range(last_max_sfb_ms, max_sfb):
    #             self.reader.SkipBits(1)  # ms_used[g][sfb]

    ################################################################################
    ## Table 4.61 – Syntax of sac_extension_data()
    ################################################################################
    def sac_extension_data(self, cnt: int) -> tuple:
        data = sac_extension_data()

        data.AncType, _ = self.reader.ReadBitsAsUInt8(2)
        data.AncStart, _ = self.reader.ReadBitAsBool()
        data.AncStop, _ = self.reader.ReadBitAsBool()
        data.AncDataSegmentByte, _ = self.reader.ReadBytes(cnt - 1)

        return cnt, data

    ################################################################################
    ## Table 4.62 – Syntax of sbr_extension_data()
    ################################################################################
    def sbr_extension_data(self, cnt: int, id_aac: int, crc_flag: bool) -> tuple:
        data = sbr_extension_data()
        num_sbr_bits = 0
        err = None

        if crc_flag:
            data.Bs_sbr_crc_bits, _ = self.reader.ReadBitsAsUInt16(10)
            num_sbr_bits += 10

        num_sbr_bits += 1
        data_bits = 0

        bs_header_flag, _ = self.reader.ReadBitAsBool()
        if bs_header_flag:
            data_bits, data.Sbr_header = self.sbr_header()
            num_sbr_bits += data_bits

        if data.Sbr_header is not None:
            sfi = self.sfi - 3
            if sfi < 0:
                sfi = 0
            elif sfi > 8:
                sfi = 8

            err = derive_sbr_tables(data, sfi, data.Sbr_header.Bs_start_freq, data.Sbr_header.Bs_stop_freq,
                                    data.Sbr_header.Bs_freq_scale, data.Sbr_header.Bs_alter_scale, 
                                    data.Sbr_header.Bs_xover_band)
            if err != None:
                return 0, data, err

            data_bits, data.Sbr_data, err = self.sbr_data(data, id_aac, data.Sbr_header.Bs_amp_res)
            num_sbr_bits += data_bits

        num_align_bits = (8 * cnt - 4 - num_sbr_bits)

        if 8 * cnt < (4 + num_sbr_bits):
            return 0, data, ValueError("sbr extension payload malformed")

        data.Bs_fill_bits, _ = self.reader.ReadBitsToByteArray(num_align_bits)

        return (num_sbr_bits + num_align_bits + 4) // 8, data, err

    ################################################################################
    ## Table 4.63 – Syntax of sbr_header()
    ################################################################################
    def sbr_header(self) -> tuple:
        data = sbr_header()
        start_bits = self.reader.BitsLeft()

        data.Bs_amp_res, _ = self.reader.ReadBitAsBool()
        data.Bs_start_freq, _ = self.reader.ReadBitsAsUInt8(4)
        data.Bs_stop_freq, _ = self.reader.ReadBitsAsUInt8(4)
        data.Bs_xover_band, _ = self.reader.ReadBitsAsUInt8(3)
        data.Bs_reserved, _ = self.reader.ReadBitsAsUInt8(2)

        data.Bs_header_extra_1, _ = self.reader.ReadBitAsBool()
        data.Bs_header_extra_2, _ = self.reader.ReadBitAsBool()
        if data.Bs_header_extra_1:
            data.Bs_freq_scale, _ = self.reader.ReadBitsAsUInt8(2)
            data.Bs_alter_scale, _ = self.reader.ReadBitsAsUInt8(1)
            data.Bs_noise_bands, _ = self.reader.ReadBitsAsUInt8(2)
        else:
            data.Bs_freq_scale = 2
            data.Bs_alter_scale = 1
            data.Bs_noise_bands = 2

        if data.Bs_header_extra_2:
            data.Bs_limiter_bands, _ = self.reader.ReadBitsAsUInt8(2)
            data.Bs_limiter_gains, _ = self.reader.ReadBitsAsUInt8(2)
            data.Bs_interpol_freq, _ = self.reader.ReadBitsAsUInt8(1)
            data.Bs_smoothing_mode, _ = self.reader.ReadBitsAsUInt8(1)
        else:
            data.Bs_limiter_bands = 2
            data.Bs_limiter_gains = 2
            data.Bs_interpol_freq = 1
            data.Bs_smoothing_mode = 1

        return (start_bits - self.reader.BitsLeft()), data

    ################################################################################
    ## Table 4.64 – Syntax of sbr_data()
    ################################################################################
    def sbr_data(self, ext_data: 'sbr_extension_data', id_aac: int, bs_amp_res: bool) -> tuple:
        data = sbr_data()
        data_bits = self.reader.BitsLeft()
        if id_aac == ID_SCE:
            data.Sbr_single_channel_element, err = self.sbr_single_channel_element(ext_data, bs_amp_res)
        elif id_aac == ID_CPE:
            data.Sbr_channel_pair_element, err = self.sbr_channel_pair_element(ext_data, bs_amp_res)
        else:
            err = None
        return (data_bits - self.reader.BitsLeft()), data, err

    ################################################################################
    ## Table 4.65 – Syntax of sbr_single_channel_element()
    ################################################################################
    def sbr_single_channel_element(self, ext_data: 'sbr_extension_data', bs_amp_res: bool) -> tuple:
        e = sbr_single_channel_element()

        e.Bs_data_extra, _ = self.reader.ReadBitAsBool()
        if e.Bs_data_extra:
            e.Bs_reserved = self.reader.ReadBitsAsUInt8(4)

        e.Sbr_grid = sbr_grid(
            Bs_var_bord_0=[0] * 2,
            Bs_var_bord_1=[0] * 2,
            Bs_num_rel_0=[0] * 2,
            Bs_num_rel_1=[0] * 2,
            bs_rel_bord_0=[[] for _ in range(2)],
            bs_rel_bord_1=[[] for _ in range(2)]
        )
        e.Sbr_dtdf = sbr_dtdf()
        e.Sbr_invf = sbr_invf()
        e.Sbr_envelope = sbr_envelope()
        e.Sbr_noise = sbr_noise()

        if err := self.sbr_grid(0, e.Sbr_grid, ext_data.Sbr_header):
            return e, err

        self.sbr_dtdf(0, e.Sbr_dtdf, e.Sbr_grid)
        self.sbr_invf(0, e.Sbr_invf, ext_data)
        self.sbr_envelope(0, False, bs_amp_res, e.Sbr_envelope, ext_data, e.Sbr_grid, e.Sbr_dtdf)
        self.sbr_noise(0, False, e.Sbr_noise, ext_data, e.Sbr_grid, e.Sbr_dtdf)

        e.Bs_add_harmonic_flag, _  = self.reader.ReadBitAsBool()
        if e.Bs_add_harmonic_flag:
            e.Sbr_sinusoidal_coding = sbr_sinusoidal_coding()
            self.sbr_sinusoidal_coding(0, e.Sbr_sinusoidal_coding, ext_data)

        e.Bs_extended_data, _  = self.reader.ReadBitAsBool()
        if e.Bs_extended_data:
            e.Bs_extension_size = self.reader.ReadBitsAsUInt8(4)
            cnt = e.Bs_extension_size
            if cnt == 15:
                e.Bs_esc_count = self.reader.ReadBitsAsUInt8(8)
                cnt += e.Bs_esc_count

            num_bits_left = cnt * 8
            if num_bits_left > MaxBitsLeft:
                return e, ValueError("Too many bits left, check bitstream continuity")

            e.Bs_extension_id = []
            e.Sbr_extension = []
            i = 0
            while num_bits_left > 7:
                ext_id = self.reader.ReadBitsAsUInt8(2)
                e.Bs_extension_id.append(ext_id)
                if e.Bs_extension_id[i] == EXTENSION_ID_PS:
                    num_bits_left -= 2

                    bits_read, ext, err = self.sbr_extension(e.Bs_extension_id[i], num_bits_left)
                    if err:
                        return e, err
                    e.Sbr_extension.append(ext)

                    num_bits_left -= bits_read
                    if num_bits_left < 0:
                        return e, ValueError("Error: SBR parsing overran available bits")

                i += 1

            e.Bs_fill_bits = self.reader.ReadBitsToByteArray(num_bits_left)

        return e, None

    ################################################################################
    ## Table 4.66 – Syntax of sbr_channel_pair_element()
    ################################################################################
    def sbr_channel_pair_element(self, ext_data: 'sbr_extension_data', bs_amp_res: bool) -> tuple:
        e = sbr_channel_pair_element()

        e.Bs_data_extra, _ = self.reader.ReadBitAsBool()
        if e.Bs_data_extra:
            e.Bs_reserved_0 = self.reader.ReadBitsAsUInt8(4)
            e.Bs_reserved_1 = self.reader.ReadBitsAsUInt8(4)

        e.Sbr_grid = sbr_grid(
            Bs_var_bord_0=[0] * 2,
            Bs_var_bord_1=[0] * 2,
            Bs_num_rel_0=[0] * 2,
            Bs_num_rel_1=[0] * 2,
            bs_rel_bord_0=[[] for _ in range(2)],
            bs_rel_bord_1=[[] for _ in range(2)]
        )
        e.Sbr_dtdf = sbr_dtdf()
        e.Sbr_invf = sbr_invf()
        e.Sbr_envelope = sbr_envelope()
        e.Sbr_noise = sbr_noise()

        e.Bs_coupling, _  = self.reader.ReadBitAsBool()
        if e.Bs_coupling:
            if err := self.sbr_grid(0, e.Sbr_grid, ext_data.Sbr_header):
                return e, err
            grid_copy(e.Sbr_grid)

            self.sbr_dtdf(0, e.Sbr_dtdf, e.Sbr_grid)
            self.sbr_dtdf(1, e.Sbr_dtdf, e.Sbr_grid)
            self.sbr_invf(0, e.Sbr_invf, ext_data)

            self.sbr_envelope(0, True, bs_amp_res, e.Sbr_envelope, ext_data, e.Sbr_grid, e.Sbr_dtdf)
            self.sbr_noise(0, True, e.Sbr_noise, ext_data, e.Sbr_grid, e.Sbr_dtdf)
            self.sbr_envelope(1, True, bs_amp_res, e.Sbr_envelope, ext_data, e.Sbr_grid, e.Sbr_dtdf)
            self.sbr_noise(1, True, e.Sbr_noise, ext_data, e.Sbr_grid, e.Sbr_dtdf)
        else:
            if err := self.sbr_grid(0, e.Sbr_grid, ext_data.Sbr_header):
                return e, err
            if err := self.sbr_grid(1, e.Sbr_grid, ext_data.Sbr_header):
                return e, err

            self.sbr_dtdf(0, e.Sbr_dtdf, e.Sbr_grid)
            self.sbr_dtdf(1, e.Sbr_dtdf, e.Sbr_grid)
            self.sbr_invf(0, e.Sbr_invf, ext_data)
            self.sbr_invf(1, e.Sbr_invf, ext_data)

            self.sbr_envelope(0, False, bs_amp_res, e.Sbr_envelope, ext_data, e.Sbr_grid, e.Sbr_dtdf)
            self.sbr_envelope(1, False, bs_amp_res, e.Sbr_envelope, ext_data, e.Sbr_grid, e.Sbr_dtdf)
            self.sbr_noise(0, False, e.Sbr_noise, ext_data, e.Sbr_grid, e.Sbr_dtdf)
            self.sbr_noise(1, False, e.Sbr_noise, ext_data, e.Sbr_grid, e.Sbr_dtdf)

        flag = self.reader.ReadBitAsBool()
        e.Bs_add_harmonic_flag.append(flag)
        e.Sbr_sinusoidal_coding = sbr_sinusoidal_coding()

        if e.Bs_add_harmonic_flag[0]:
            self.sbr_sinusoidal_coding(0, e.Sbr_sinusoidal_coding, ext_data)
        else:
            e.Sbr_sinusoidal_coding.Bs_add_harmonic = [[] for _ in range(1)]

        flag = self.reader.ReadBitAsBool()
        e.Bs_add_harmonic_flag.append(flag)
        if e.Bs_add_harmonic_flag[1]:
            self.sbr_sinusoidal_coding(1, e.Sbr_sinusoidal_coding, ext_data)

        e.Bs_extended_data, _ = self.reader.ReadBitAsBool()
        if e.Bs_extended_data:
            e.Bs_extension_size = self.reader.ReadBitsAsUInt8(4)
            cnt = e.Bs_extension_size
            if cnt == 15:
                e.Bs_esc_count = self.reader.ReadBitsAsUInt8(8)
                cnt += e.Bs_esc_count

            num_bits_left = cnt * 8

            if num_bits_left > MaxBitsLeft:
                return e, ValueError("Too many bits left, check bitstream continuity")

            self.reader.SkipBits(num_bits_left)

        return e, None

    ################################################################################
    ## Table 4.67 – Syntax of sbr_channel_pair_base_element()
    ################################################################################
    def sbr_channel_pair_base_element(self, bs_amp_res: bool, ext_data: 'sbr_extension_data') -> tuple:
        e = sbr_channel_pair_base_element()

        e.Bs_data_extra = self.reader.ReadBitAsBool()
        if e.Bs_data_extra:
            e.Bs_reserved_0 = self.reader.ReadBitsAsUInt8(4)
            e.Bs_reserved_1 = self.reader.ReadBitsAsUInt8(4)

        e.Bs_coupling = self.reader.ReadBitAsBool()

        e.Sbr_grid = sbr_grid(
            Bs_var_bord_0=[0] * 2,
            Bs_var_bord_1=[0] * 2,
            Bs_num_rel_0=[0] * 2,
            Bs_num_rel_1=[0] * 2,
            bs_rel_bord_0=[[] for _ in range(2)],
            bs_rel_bord_1=[[] for _ in range(2)]
        )
        e.Sbr_dtdf = sbr_dtdf()
        e.Sbr_invf = sbr_invf()
        e.Sbr_envelope = sbr_envelope()
        e.Sbr_noise = sbr_noise()

        if err := self.sbr_grid(0, e.Sbr_grid, ext_data.Sbr_header):
            return e, err

        self.sbr_dtdf(0, e.Sbr_dtdf, e.Sbr_grid)
        self.sbr_invf(0, e.Sbr_invf, ext_data)
        self.sbr_envelope(0, True, bs_amp_res, e.Sbr_envelope, ext_data, e.Sbr_grid, e.Sbr_dtdf)
        self.sbr_noise(0, True, e.Sbr_noise, ext_data, e.Sbr_grid, e.Sbr_dtdf)

        e.Bs_add_harmonic_flag = self.reader.ReadBitAsBool()
        if e.Bs_add_harmonic_flag:
            e.Sbr_sinusoidal_coding = sbr_sinusoidal_coding()
            self.sbr_sinusoidal_coding(0, e.Sbr_sinusoidal_coding, ext_data)

        e.Bs_extended_data = self.reader.ReadBitAsBool()
        if e.Bs_extended_data:
            e.Bs_extension_size = self.reader.ReadBitsAsUInt8(4)
            cnt = e.Bs_extension_size
            if cnt == 15:
                e.Bs_esc_count = self.reader.ReadBitsAsUInt8(8)
                cnt += e.Bs_esc_count

            num_bits_left = cnt * 8
            if num_bits_left > MaxBitsLeft:
                return e, ValueError("Too many bits left, check bitstream continuity")

            e.Bs_extension_id = []
            e.Sbr_extension = []
            for i in range(2):
                ext_id = self.reader.ReadBitsAsUInt8(2)
                e.Bs_extension_id.append(ext_id)
                if e.Bs_extension_id[i] == EXTENSION_ID_PS:
                    num_bits_left -= 2

                    bits_read, ext, err = self.sbr_extension(e.Bs_extension_id[i], num_bits_left)
                    if err:
                        return e, err
                    e.Sbr_extension.append(ext)

                    num_bits_left -= bits_read
                    if num_bits_left < 0:
                        return e, ValueError("Error: SBR parsing overran available bits")

            e.Bs_fill_bits = self.reader.ReadBitsToByteArray(num_bits_left)

        return e, None

    ################################################################################
    ## Table 4.68 – Syntax of sbr_channel_pair_enhance_element()
    ################################################################################
    def sbr_channel_pair_enhance_element(self, bs_amp_res: bool) -> 'sbr_channel_pair_enhance_element':
        # e = sbr_channel_pair_enhance_element()
        #
        # e.Sbr_dtdf = self.sbr_dtdf(1, ???)
        # e.Sbr_envelope = self.sbr_envelope(1, 1, bs_amp_res, None, e.Sbr_dtdf)
        # e.Sbr_noise = self.sbr_noise(1, 1)
        #
        # if e.Bs_add_harmonic_flag := self.reader.ReadBitAsBool():
        #     e.Sbr_sinusoidal_coding = self.sbr_sinusoidal_coding(1)

        return None

    ################################################################################
    ## Table 4.69 – Syntax of sbr_grid()
    ################################################################################
    def sbr_grid(self, ch: int, data: 'sbr_grid', header: 'sbr_header') -> Optional[Exception]:
        data.Bs_frame_class[ch], _ = self.reader.ReadBitsAsUInt8(2)
        if data.Bs_frame_class[ch] == FIXFIX:
            data.Tmp, _ = self.reader.ReadBitsAsUInt8(2)
            data.bs_num_env.append(1 << data.Tmp)

            if data.bs_num_env[ch] == 1:
                header.Bs_amp_res = False

            data.Bs_freq_res.append([0] * data.bs_num_env[ch])
            data.Bs_freq_res[ch][0], _ = self.reader.ReadBitsAsUInt8(1)
            for env in range(1, data.bs_num_env[ch]):
                data.Bs_freq_res[ch][env] = data.Bs_freq_res[ch][0]

            data.Bs_pointer.append(0)

        elif data.Bs_frame_class[ch] == FIXVAR:
            tmp, _ = self.reader.ReadBitsAsUInt8(2)
            data.Bs_var_bord_1[ch] = tmp
            tmp, _ = self.reader.ReadBitsAsUInt8(2)
            data.Bs_num_rel_1[ch] = tmp
            data.bs_num_env.append(data.Bs_num_rel_1[ch] + 1)

            data.bs_rel_bord_1[ch] = [0] * (data.bs_num_env[ch] - 1)
            for rel in range(len(data.bs_rel_bord_1[ch])):
                data.Tmp, _ = self.reader.ReadBitsAsUInt8(2)
                data.bs_rel_bord_1[ch][rel] = 2 * data.Tmp + 1

            ptr_bits = ceil_log2(data.bs_num_env[ch] + 1)
            ptr, _ = self.reader.ReadBitsAsUInt(ptr_bits)
            data.Bs_pointer.append(ptr)

            data.Bs_freq_res.append([0] * data.bs_num_env[ch])
            for env in range(len(data.Bs_freq_res[ch])):
                data.Bs_freq_res[ch][data.bs_num_env[ch] - 1 - env], _ = self.reader.ReadBitsAsUInt8(1)

        elif data.Bs_frame_class[ch] == VARFIX:
            tmp, _ = self.reader.ReadBitsAsUInt8(2)
            data.Bs_var_bord_0[ch] = tmp
            tmp, _ = self.reader.ReadBitsAsUInt8(2)
            data.Bs_num_rel_0[ch] = tmp
            data.bs_num_env.append(data.Bs_num_rel_0[ch] + 1)

            data.bs_rel_bord_0[ch] = [0] * (data.bs_num_env[ch] - 1)
            for rel in range(len(data.bs_rel_bord_0[ch])):
                data.Tmp, _ = self.reader.ReadBitsAsUInt8(2)
                data.bs_rel_bord_0[ch][rel] = 2 * data.Tmp + 2

            ptr_bits = ceil_log2(data.bs_num_env[ch] + 1)
            ptr, _ = self.reader.ReadBitsAsUInt(ptr_bits)
            data.Bs_pointer.append(ptr)

            data.Bs_freq_res.append([0] * data.bs_num_env[ch])
            for env in range(len(data.Bs_freq_res[ch])):
                data.Bs_freq_res[ch][env], _ = self.reader.ReadBitsAsUInt8(1)

        elif data.Bs_frame_class[ch] == VARVAR:
            tmp, _ = self.reader.ReadBitsAsUInt8(2)
            data.Bs_var_bord_0[ch] = tmp
            tmp, _ = self.reader.ReadBitsAsUInt8(2)
            data.Bs_var_bord_1[ch] = tmp
            tmp, _ = self.reader.ReadBitsAsUInt8(2)
            data.Bs_num_rel_0[ch] = tmp
            tmp, _ = self.reader.ReadBitsAsUInt8(2)
            data.Bs_num_rel_1[ch] = tmp

            data.bs_num_env.append(min(5, data.Bs_num_rel_0[ch] + data.Bs_num_rel_1[ch] + 1))

            data.bs_rel_bord_0[ch] = [0] * data.Bs_num_rel_0[ch]
            for rel in range(len(data.bs_rel_bord_0[ch])):
                data.Tmp, _ = self.reader.ReadBitsAsUInt8(2)
                data.bs_rel_bord_0[ch][rel] = 2 * data.Tmp + 2

            data.bs_rel_bord_1[ch] = [0] * data.Bs_num_rel_1[ch]
            for rel in range(len(data.bs_rel_bord_1[ch])):
                data.Tmp, _ = self.reader.ReadBitsAsUInt8(2)
                data.bs_rel_bord_1[ch][rel] = 2 * data.Tmp + 2

            ptr_bits = ceil_log2(data.bs_num_env[ch] + 1)
            ptr, _ = self.reader.ReadBitsAsUInt(ptr_bits)
            data.Bs_pointer.append(ptr)

            data.Bs_freq_res.append([0] * data.bs_num_env[ch])
            for env in range(len(data.Bs_freq_res[ch])):
                data.Bs_freq_res[ch][env], _ = self.reader.ReadBitsAsUInt8(1)

        if data.bs_num_env[ch] > 1:
            data.bs_num_noise.append(2)
        elif data.bs_num_env[ch] == 1:
            data.bs_num_noise.append(1)
        else:
            return ValueError(f"Error: bs_num_env[{ch}] ({data.bs_num_env[ch]}) is out of range")

        return None

    ################################################################################
    ## Table 4.70 – Syntax of sbr_dtdf()
    ################################################################################
    def sbr_dtdf(self, ch: int, data: 'sbr_dtdf', grid: 'sbr_grid'):
        data.Bs_df_env.append([False] * grid.bs_num_env[ch])
        for env in range(len(data.Bs_df_env[ch])):
            data.Bs_df_env[ch][env], _ = self.reader.ReadBitAsBool()

        data.Bs_df_noise.append([False] * grid.bs_num_noise[ch])
        for noise in range(len(data.Bs_df_noise[ch])):
            data.Bs_df_noise[ch][noise], _ = self.reader.ReadBitAsBool()

    ################################################################################
    ## Table 4.71 – Syntax of sbr_invf()
    ################################################################################
    def sbr_invf(self, ch: int, data: 'sbr_invf', ext_data: 'sbr_extension_data'):
        data.Bs_invf_mode.append([0] * ext_data.N_Q)
        for n in range(len(data.Bs_invf_mode[ch])):
            data.Bs_invf_mode[ch][n], _ = self.reader.ReadBitsAsUInt8(2)

    ################################################################################
    ## Table 4.72 – Syntax of sbr_envelope()
    ################################################################################
    def sbr_envelope(
        self, ch: int, bs_coupling: bool, bs_amp_res: bool, 
        e: 'sbr_envelope', ext_data: 'sbr_extension_data', grid: 'sbr_grid', dtdf: 'sbr_dtdf'
    ):
        t_huff = []
        f_huff = []

        amp_res = bs_amp_res
        if grid.bs_num_env[ch] == 1 and grid.Bs_frame_class[ch] == FIXFIX:
            amp_res = False

        if bs_coupling and ch == 1:
            if amp_res:
                t_huff = t_huffman_env_bal_3_0dB
                f_huff = f_huffman_env_bal_3_0dB
            else:
                t_huff = t_huffman_env_bal_1_5dB
                f_huff = f_huffman_env_bal_1_5dB
        else:
            if amp_res:
                t_huff = t_huffman_env_3_0dB
                f_huff = f_huffman_env_3_0dB
            else:
                t_huff = t_huffman_env_1_5dB
                f_huff = f_huffman_env_1_5dB

        e.Bs_data_env.append([[] for _ in range(grid.bs_num_env[ch])])
        for env in range(len(e.Bs_data_env[ch])):
            if not dtdf.Bs_df_env[ch][env]:
                num_bands = ext_data.n[grid.Bs_freq_res[ch][env]]
                e.Bs_data_env[ch][env] = [0] * num_bands
                if bs_coupling and ch == 1:
                    if amp_res:
                        e.Bs_env_start_value_balance, _ = self.reader.ReadBitsAsUInt8(5)
                        e.Bs_data_env[ch][env][0] = int(e.Bs_env_start_value_balance)
                    else:
                        e.Bs_env_start_value_balance, _ = self.reader.ReadBitsAsUInt8(6)
                        e.Bs_data_env[ch][env][0] = int(e.Bs_env_start_value_balance)
                else:
                    if amp_res:
                        e.Bs_env_start_value_level, _ = self.reader.ReadBitsAsUInt8(6)
                        e.Bs_data_env[ch][env][0] = int(e.Bs_env_start_value_level)
                    else:
                        e.Bs_env_start_value_level, _ = self.reader.ReadBitsAsUInt8(7)
                        e.Bs_data_env[ch][env][0] = int(e.Bs_env_start_value_level)

                for band in range(1, num_bands):
                    e.Bs_data_env[ch][env][band] = sbr_huff_dec(self.reader, f_huff)
            else:
                num_bands = ext_data.n[grid.Bs_freq_res[ch][env]]
                e.Bs_data_env[ch][env] = [0] * num_bands
                for band in range(num_bands):
                    e.Bs_data_env[ch][env][band] = sbr_huff_dec(self.reader, t_huff)

    ################################################################################
    ## Table 4.73 – Syntax of sbr_noise()
    ################################################################################
    def sbr_noise(
        self, ch: int, bs_coupling: bool, 
        data: 'sbr_noise', ext_data: 'sbr_extension_data', grid: 'sbr_grid', dtdf: 'sbr_dtdf'
    ):
        t_huff = []
        f_huff = []

        if bs_coupling and ch == 1:
            t_huff = t_huffman_noise_bal_3_0dB
            f_huff = f_huffman_env_bal_3_0dB
        else:
            t_huff = t_huffman_noise_3_0dB
            f_huff = f_huffman_env_3_0dB

        data.Bs_data_noise.append([[] for _ in range(grid.bs_num_noise[ch])])
        for noise in range(len(data.Bs_data_noise[ch])):
            data.Bs_data_noise[ch][noise] = [0] * ext_data.N_Q
            if not dtdf.Bs_df_noise[ch][noise]:
                if bs_coupling and ch == 1:
                    data.Bs_noise_start_value_balance, _ = self.reader.ReadBitsAsUInt8(5)
                    data.Bs_data_noise[ch][noise][0] = int(data.Bs_noise_start_value_balance)
                else:
                    data.Bs_noise_start_value_level, _ = self.reader.ReadBitsAsUInt8(5)
                    data.Bs_data_noise[ch][noise][0] = int(data.Bs_noise_start_value_level)

                for band in range(1, ext_data.N_Q):
                    data.Bs_data_noise[ch][noise][band] = sbr_huff_dec(self.reader, f_huff)
            else:
                for band in range(len(data.Bs_data_noise[ch][noise])):
                    data.Bs_data_noise[ch][noise][band] = sbr_huff_dec(self.reader, t_huff)

    ################################################################################
    ## Table 4.74 – Syntax of sbr_sinusoidal_coding()
    ################################################################################
    def sbr_sinusoidal_coding(self, ch: int, data: 'sbr_sinusoidal_coding', ext_data: 'sbr_extension_data'):
        data.Bs_add_harmonic.append([False] * ext_data.N_high)
        for n in range(len(data.Bs_add_harmonic[ch])):
            data.Bs_add_harmonic[ch][n], _ = self.reader.ReadBitAsBool()

    ################################################################################
    ## Table 8.A.1 – Syntax of sbr_extension()
    ################################################################################
    def sbr_extension(self, bs_extension_id: int, num_bits_left: int) -> (int, 'sbr_extension', Exception):
        data = sbr_extension()
        if bs_extension_id == EXTENSION_ID_PS:
            # num_bits_left -= ps_data()
            return num_bits_left, None, Exception("bs_extension_id of 2 (EXTENSION_ID_PS) unsupported")
        else:
            # data.Bs_fill_bits, _ = adts.reader.ReadBitsToByteArray(num_bits_left)
            pass
        
        return num_bits_left, data, None

    def Debug(self) -> str:
        return "MPEGVer: {}, Layer: {}, Profile: {}, SampFreq: {}, ChannelConfig: {}, Bitrate: {}".format(
            self.MpegVersion,
            self.Layer,
            AACProfileType[self.Profile],
            self.SamplingFrequency,
            ChannelConfiguration[self.ChannelConfiguration],
            self.Bitrate
        )

class single_channel_element:
    def __init__(self, 
                 Element_instance_tag: int = 0, 
                 Channel_stream: 'individual_channel_stream' = None):
        self.Element_instance_tag = Element_instance_tag
        self.Channel_stream = Channel_stream if Channel_stream is not None else individual_channel_stream()

class channel_pair_element:
    def __init__(self, 
                 Element_instance_tag: int = 0, 
                 Common_window: bool = False, 
                 Ics_info: 'ics_info' = None, 
                 Ms_used: list = None, 
                 Channel_stream1: 'individual_channel_stream' = None, 
                 Channel_stream2: 'individual_channel_stream' = None):
        self.Element_instance_tag = Element_instance_tag
        self.Common_window = Common_window
        self.Ics_info = Ics_info if Ics_info is not None else ics_info()
        self.Ms_used = Ms_used if Ms_used is not None else [[False]]
        self.Channel_stream1 = Channel_stream1 if Channel_stream1 is not None else individual_channel_stream()
        self.Channel_stream2 = Channel_stream2 if Channel_stream2 is not None else individual_channel_stream()

class coupling_channel_element:
    def __init__(self, 
                 Element_instance_tag: int = 0, 
                 Ind_sw_cce_flag: bool = False, 
                 Num_coupled_elements: int = 0, 
                 Cc_target_is_cpe: list = None, 
                 Cc_target_tag_select: list = None, 
                 Cc_l: list = None, 
                 Cc_r: list = None, 
                 Cc_domain: bool = False, 
                 Gain_element_sign: bool = False, 
                 Gain_element_scale: int = 0, 
                 Channel_stream: 'individual_channel_stream' = None, 
                 Common_gain_element_present: list = None, 
                 Common_gain_element: list = None, 
                 dpcm_gain_element: list = None):
        self.Element_instance_tag = Element_instance_tag
        self.Ind_sw_cce_flag = Ind_sw_cce_flag
        self.Num_coupled_elements = Num_coupled_elements
        self.Cc_target_is_cpe = Cc_target_is_cpe if Cc_target_is_cpe is not None else []
        self.Cc_target_tag_select = Cc_target_tag_select if Cc_target_tag_select is not None else []
        self.Cc_l = Cc_l if Cc_l is not None else []
        self.Cc_r = Cc_r if Cc_r is not None else []
        self.Cc_domain = Cc_domain
        self.Gain_element_sign = Gain_element_sign
        self.Gain_element_scale = Gain_element_scale

        self.Channel_stream = Channel_stream if Channel_stream is not None else individual_channel_stream()

        self.Common_gain_element_present = Common_gain_element_present if Common_gain_element_present is not None else []
        self.Common_gain_element = Common_gain_element if Common_gain_element is not None else []

        self.dpcm_gain_element = dpcm_gain_element if dpcm_gain_element is not None else [[[0]]]

class lfe_channel_element:
    def __init__(self, 
                 Element_instance_tag: int = 0, 
                 Channel_stream: 'individual_channel_stream' = None):
        self.Element_instance_tag = Element_instance_tag
        self.Channel_stream = Channel_stream if Channel_stream is not None else individual_channel_stream()

class data_stream_element:
    def __init__(self, 
                 Element_instance_tag: int = 0, 
                 Data_byte_align_flag: bool = False, 
                 Count: int = 0, 
                 Esc_count: int = 0, 
                 Data_stream_byte: list = None):
        self.Element_instance_tag = Element_instance_tag
        self.Data_byte_align_flag = Data_byte_align_flag
        self.Count = Count
        self.Esc_count = Esc_count
        self.Data_stream_byte = Data_stream_byte if Data_stream_byte is not None else [[0]]

class program_config_element:
    def __init__(self, 
                 Element_instance_tag: int = 0,
                 Object_type: int = 0,
                 Sampling_frequency_index: int = 0,
                 Num_front_channel_elements: int = 0,
                 Num_side_channel_elements: int = 0,
                 Num_back_channel_elements: int = 0,
                 Num_lfe_channel_elements: int = 0,
                 Num_assoc_data_elements: int = 0,
                 Num_valid_cc_elements: int = 0,
                 Mono_mixdown_present: bool = False,
                 Mono_mixdown_element_num: int = 0,
                 Stereo_mixdown_present: bool = False,
                 Stereo_mixdown_element_num: int = 0,
                 Matrix_mixdown_idx_present: bool = False,
                 Matrix_mixdown_idx: int = 0,
                 Pseudo_surround_enable: bool = False,
                 Front_element_is_cpe: list = None,
                 Front_element_tag_select: list = None,
                 Side_element_is_cpe: list = None,
                 Side_element_tag_select: list = None,
                 Back_element_is_cpe: list = None,
                 Back_element_tag_select: list = None,
                 Lfe_element_tag_select: list = None,
                 Assoc_data_element_tag_select: list = None,
                 Cc_element_is_ind_sw: list = None,
                 Valid_cc_element_tag_select: list = None,
                 Comment_field_bytes: int = 0,
                 Comment_field_data: bytes = None):
        
        self.Element_instance_tag = Element_instance_tag
        self.Object_type = Object_type
        self.Sampling_frequency_index = Sampling_frequency_index
        self.Num_front_channel_elements = Num_front_channel_elements
        self.Num_side_channel_elements = Num_side_channel_elements
        self.Num_back_channel_elements = Num_back_channel_elements
        self.Num_lfe_channel_elements = Num_lfe_channel_elements
        self.Num_assoc_data_elements = Num_assoc_data_elements
        self.Num_valid_cc_elements = Num_valid_cc_elements

        self.Mono_mixdown_present = Mono_mixdown_present
        self.Mono_mixdown_element_num = Mono_mixdown_element_num

        self.Stereo_mixdown_present = Stereo_mixdown_present
        self.Stereo_mixdown_element_num = Stereo_mixdown_element_num

        self.Matrix_mixdown_idx_present = Matrix_mixdown_idx_present
        self.Matrix_mixdown_idx = Matrix_mixdown_idx
        self.Pseudo_surround_enable = Pseudo_surround_enable

        self.Front_element_is_cpe = Front_element_is_cpe if Front_element_is_cpe is not None else []
        self.Front_element_tag_select = Front_element_tag_select if Front_element_tag_select is not None else []

        self.Side_element_is_cpe = Side_element_is_cpe if Side_element_is_cpe is not None else []
        self.Side_element_tag_select = Side_element_tag_select if Side_element_tag_select is not None else []

        self.Back_element_is_cpe = Back_element_is_cpe if Back_element_is_cpe is not None else []
        self.Back_element_tag_select = Back_element_tag_select if Back_element_tag_select is not None else []

        self.Lfe_element_tag_select = Lfe_element_tag_select if Lfe_element_tag_select is not None else []
        self.Assoc_data_element_tag_select = Assoc_data_element_tag_select if Assoc_data_element_tag_select is not None else []

        self.Cc_element_is_ind_sw = Cc_element_is_ind_sw if Cc_element_is_ind_sw is not None else []
        self.Valid_cc_element_tag_select = Valid_cc_element_tag_select if Valid_cc_element_tag_select is not None else []

        self.Comment_field_bytes = Comment_field_bytes
        self.Comment_field_data = Comment_field_data if Comment_field_data is not None else b''

class fill_element:
    def __init__(self, 
                 Count: int = 0, 
                 Esc_count: int = 0, 
                 Extension_payload: 'extension_payload' = None):
        self.Count = Count
        self.Esc_count = Esc_count
        self.Extension_payload = Extension_payload if Extension_payload is not None else extension_payload()

class adts_error_check:
    def __init__(self, Crc_check: int = 0):
        self.Crc_check = Crc_check


class adts_header_error_check:
    def __init__(self, 
                 Raw_data_block_position: list = None, 
                 Crc_check: int = 0):
        self.Raw_data_block_position = Raw_data_block_position if Raw_data_block_position is not None else []
        self.Crc_check = Crc_check

class adts_raw_data_block_error_check:
    def __init__(self, Crc_check: int = 0):
        self.Crc_check = Crc_check

class dynamic_range_info:
    def __init__(self, 
                 Pce_tag_present: bool = False, 
                 Pce_instance_tag: int = 0, 
                 Drc_tag_reserve_bits: int = 0, 
                 Excluded_chns_present: bool = False, 
                 Excluded_chns: 'excluded_channels' = None, 
                 Drc_bands_present: bool = False, 
                 Drc_band_incr: int = 0, 
                 Drc_interpolation_scheme: int = 0, 
                 Drc_band_top: bytes = None, 
                 Prog_ref_level_present: bool = False, 
                 Prog_ref_level: int = 0, 
                 Prog_ref_level_reserved_bits: int = 0, 
                 Dyn_range_sign: list = None, 
                 Dyn_range_cnt: list = None):
        self.Pce_tag_present = Pce_tag_present
        self.Pce_instance_tag = Pce_instance_tag
        self.Drc_tag_reserve_bits = Drc_tag_reserve_bits

        self.Excluded_chns_present = Excluded_chns_present
        self.Excluded_chns = Excluded_chns if Excluded_chns is not None else excluded_channels()

        self.Drc_bands_present = Drc_bands_present
        self.Drc_band_incr = Drc_band_incr
        self.Drc_interpolation_scheme = Drc_interpolation_scheme
        self.Drc_band_top = Drc_band_top if Drc_band_top is not None else b''

        self.Prog_ref_level_present = Prog_ref_level_present
        self.Prog_ref_level = Prog_ref_level
        self.Prog_ref_level_reserved_bits = Prog_ref_level_reserved_bits

        self.Dyn_range_sign = Dyn_range_sign if Dyn_range_sign is not None else []
        self.Dyn_range_cnt = Dyn_range_cnt if Dyn_range_cnt is not None else []

class excluded_channels:
    def __init__(self, 
                 Exclude_mask: list = None, 
                 Additional_excluded_chns: list = None):
        self.Exclude_mask = Exclude_mask if Exclude_mask is not None else []
        self.Additional_excluded_chns = Additional_excluded_chns if Additional_excluded_chns is not None else []

class extension_payload:
    def __init__(self, 
                 Extension_type: int = 0, 
                 Fill_nibble: int = 0, 
                 Fill_byte: bytes = None, 
                 Data_element_version: int = 0, 
                 DataElementLengthPart: int = 0,  
                 Data_element_byte: bytes = None, 
                 Dynamic_range_info: 'dynamic_range_info' = None, 
                 Sac_extension_data: 'sac_extension_data' = None, 
                 Sbr_extension_data: 'sbr_extension_data' = None, 
                 Other_bits: list = None):
        self.Extension_type = Extension_type
        self.Fill_nibble = Fill_nibble
        self.Fill_byte = Fill_byte if Fill_byte is not None else b''

        self.Data_element_version = Data_element_version
        self.DataElementLengthPart = DataElementLengthPart
        self.Data_element_byte = Data_element_byte if Data_element_byte is not None else b''

        self.Dynamic_range_info = Dynamic_range_info if Dynamic_range_info is not None else dynamic_range_info()
        self.Sac_extension_data = Sac_extension_data if Sac_extension_data is not None else sac_extension_data()
        self.Sbr_extension_data = Sbr_extension_data if Sbr_extension_data is not None else sbr_extension_data()

        self.Other_bits = Other_bits if Other_bits is not None else []

class gain_control_data:
    def __init__(self, 
                 Max_band: int = 0, 
                 Alevcode: list = None, 
                 Aloccode: list = None, 
                 Adjust_num: list = None):
        self.Max_band = Max_band
        self.Alevcode = Alevcode if Alevcode is not None else [[[0]]]
        self.Aloccode = Aloccode if Aloccode is not None else [[[0]]]
        self.Adjust_num = Adjust_num if Adjust_num is not None else [[0]]

class individual_channel_stream:
    def __init__(self, 
                 Global_gain: int = 0, 
                 Ics_info: 'ics_info' = None, 
                 Section_data: 'section_data' = None, 
                 Scale_factor_data: 'scale_factor_data' = None, 
                 Pulse_data_present: bool = False, 
                 Pulse_data: 'pulse_data' = None, 
                 Tns_data_present: bool = False, 
                 Tns_data: 'tns_data' = None, 
                 Gain_control_data_present: bool = False, 
                 Gain_control_data: 'gain_control_data' = None, 
                 Spectral_data: 'spectral_data' = None, 
                 Length_of_reordered_spectral_data: int = 0, 
                 Length_of_longest_code_word: int = 0, 
                 Reordered_spectral_data: 'reordered_spectral_data' = None):
        
        self.Global_gain = Global_gain
        self.Ics_info = Ics_info if Ics_info is not None else ics_info()
        self.Section_data = Section_data if Section_data is not None else section_data()
        self.Scale_factor_data = Scale_factor_data if Scale_factor_data is not None else scale_factor_data()

        self.Pulse_data_present = Pulse_data_present
        self.Pulse_data = Pulse_data if Pulse_data is not None else pulse_data()

        self.Tns_data_present = Tns_data_present
        self.Tns_data = Tns_data if Tns_data is not None else tns_data()

        self.Gain_control_data_present = Gain_control_data_present
        self.Gain_control_data = Gain_control_data if Gain_control_data is not None else gain_control_data()

        self.Spectral_data = Spectral_data if Spectral_data is not None else spectral_data()

        self.Length_of_reordered_spectral_data = Length_of_reordered_spectral_data
        self.Length_of_longest_code_word = Length_of_longest_code_word
        self.Reordered_spectral_data = Reordered_spectral_data if Reordered_spectral_data is not None else reordered_spectral_data()

class ics_info:
    def __init__(self, 
                 Window_sequence: int = 0, 
                 Window_shape: int = 0, 
                 Max_sfb: int = 0, 
                 Scale_factor_grouping: int = 0, 
                 Predictor_data_present: bool = False, 
                 Predictor_reset: bool = False, 
                 Predictor_reset_group_num: int = 0, 
                 Prediction_used: list = None, 
                 num_windows: int = 0, 
                 num_window_groups: int = 0, 
                 window_group_length: list = None, 
                 sect_sfb_offset: list = None, 
                 swb_offset: list = None, 
                 sfb_cb: list = None, 
                 num_swb: int = 0, 
                 Ltp_data_present: bool = False, 
                 Ltp_data: 'ltp_data' = None):
        self.Window_sequence = Window_sequence
        self.Window_shape = Window_shape
        self.Max_sfb = Max_sfb
        self.Scale_factor_grouping = Scale_factor_grouping
        self.Predictor_data_present = Predictor_data_present
        self.Predictor_reset = Predictor_reset
        self.Predictor_reset_group_num = Predictor_reset_group_num
        self.Prediction_used = Prediction_used if Prediction_used is not None else []
        self.num_windows = num_windows
        self.num_window_groups = num_window_groups
        self.window_group_length = window_group_length if window_group_length is not None else []
        self.sect_sfb_offset = sect_sfb_offset if sect_sfb_offset is not None else []
        self.swb_offset = swb_offset if swb_offset is not None else []
        self.sfb_cb = sfb_cb if sfb_cb is not None else []
        self.num_swb = num_swb
        self.Ltp_data_present = Ltp_data_present
        self.Ltp_data = Ltp_data if Ltp_data is not None else ltp_data()

class ltp_data:
    def __init__(self, Ltp_lag: int = 0, Ltp_coef: int = 0, Ltp_long_used: list = None):
        self.Ltp_lag = Ltp_lag
        self.Ltp_coef = Ltp_coef
        self.Ltp_long_used = Ltp_long_used if Ltp_long_used is not None else []

class pulse_data:
    def __init__(self, 
                 Number_pulse: int = 0, 
                 Pulse_start_sfb: int = 0, 
                 Pulse_offset: list = None, 
                 Pulse_amp: list = None):
        self.Number_pulse = Number_pulse
        self.Pulse_start_sfb = Pulse_start_sfb
        self.Pulse_offset = Pulse_offset if Pulse_offset is not None else []
        self.Pulse_amp = Pulse_amp if Pulse_amp is not None else []

class reordered_spectral_data:
    def __init__(self, 
                 Data: list = None):
        self.Data = Data if Data is not None else []

class sac_extension_data:
    def __init__(self, 
                 AncType: int = 0,  
                 AncStart: bool = False, 
                 AncStop: bool = False, 
                 AncDataSegmentByte: bytes = None):
        self.AncType = AncType
        self.AncStart = AncStart
        self.AncStop = AncStop
        self.AncDataSegmentByte = AncDataSegmentByte if AncDataSegmentByte is not None else b''

class sbr_extension_data:
    def __init__(self, 
                 Bs_sbr_crc_bits: int = 0, 
                 Bs_header_flag: bool = False, 
                 Bs_fill_bits: list = None, 
                 Sbr_header: 'sbr_header' = None, 
                 Sbr_data: 'sbr_data' = None, 
                 num_sbr_bits: int = 0, 
                 num_align_bits: int = 0, 
                 k0: int = 0, 
                 k2: int = 0, 
                 f_master: list = None, 
                 f_tablehigh: list = None, 
                 f_tablelow: list = None, 
                 f_tablenoise: list = None, 
                 M: int = 0, 
                 k_x: int = 0, 
                 N_master: int = 0, 
                 N_high: int = 0, 
                 N_low: int = 0, 
                 n: list = None, 
                 N_Q: int = 0):
        self.Bs_sbr_crc_bits = Bs_sbr_crc_bits
        self.Bs_header_flag = Bs_header_flag
        self.Bs_fill_bits = Bs_fill_bits if Bs_fill_bits is not None else []
        self.Sbr_header = Sbr_header
        self.Sbr_data = Sbr_data
        self.num_sbr_bits = num_sbr_bits
        self.num_align_bits = num_align_bits
        self.k0 = k0
        self.k2 = k2
        self.f_master = f_master if f_master is not None else []
        self.f_tablehigh = f_tablehigh if f_tablehigh is not None else []
        self.f_tablelow = f_tablelow if f_tablelow is not None else []
        self.f_tablenoise = f_tablenoise if f_tablenoise is not None else []
        self.M = M
        self.k_x = k_x
        self.N_master = N_master
        self.N_high = N_high
        self.N_low = N_low
        self.n = n if n is not None else []
        self.N_Q = N_Q

class sbr_header:
    def __init__(self, 
                 Bs_amp_res: bool = False, 
                 Bs_start_freq: int = 0, 
                 Bs_stop_freq: int = 0, 
                 Bs_xover_band: int = 0, 
                 Bs_reserved: int = 0, 
                 Bs_header_extra_1: bool = False, 
                 Bs_header_extra_2: bool = False, 
                 Bs_freq_scale: int = 0, 
                 Bs_alter_scale: int = 0, 
                 Bs_noise_bands: int = 0, 
                 Bs_limiter_bands: int = 0, 
                 Bs_limiter_gains: int = 0, 
                 Bs_interpol_freq: int = 0, 
                 Bs_smoothing_mode: int = 0):
        self.Bs_amp_res = Bs_amp_res
        self.Bs_start_freq = Bs_start_freq
        self.Bs_stop_freq = Bs_stop_freq
        self.Bs_xover_band = Bs_xover_band
        self.Bs_reserved = Bs_reserved
        self.Bs_header_extra_1 = Bs_header_extra_1
        self.Bs_header_extra_2 = Bs_header_extra_2
        self.Bs_freq_scale = Bs_freq_scale
        self.Bs_alter_scale = Bs_alter_scale
        self.Bs_noise_bands = Bs_noise_bands
        self.Bs_limiter_bands = Bs_limiter_bands
        self.Bs_limiter_gains = Bs_limiter_gains
        self.Bs_interpol_freq = Bs_interpol_freq
        self.Bs_smoothing_mode = Bs_smoothing_mode

class sbr_data:
    def __init__(self, 
                 Sbr_single_channel_element: 'sbr_single_channel_element' = None, 
                 Sbr_channel_pair_element: 'sbr_channel_pair_element' = None, 
                 Sbr_channel_pair_base_element: 'sbr_channel_pair_base_element' = None, 
                 Sbr_channel_pair_enhance_element: 'sbr_channel_pair_enhance_element' = None):
        self.Sbr_single_channel_element = Sbr_single_channel_element if Sbr_single_channel_element is not None else sbr_single_channel_element()
        self.Sbr_channel_pair_element = Sbr_channel_pair_element if Sbr_channel_pair_element is not None else sbr_channel_pair_element()
        self.Sbr_channel_pair_base_element = Sbr_channel_pair_base_element if Sbr_channel_pair_base_element is not None else sbr_channel_pair_base_element()
        self.Sbr_channel_pair_enhance_element = Sbr_channel_pair_enhance_element if Sbr_channel_pair_enhance_element is not None else sbr_channel_pair_enhance_element()

class sbr_single_channel_element:
    def __init__(self, 
                 Bs_data_extra: bool = False, 
                 Bs_reserved: int = 0, 
                 Sbr_grid: 'sbr_grid' = None, 
                 Sbr_dtdf: 'sbr_dtdf' = None, 
                 Sbr_invf: 'sbr_invf' = None, 
                 Sbr_envelope: 'sbr_envelope' = None, 
                 Sbr_noise: 'sbr_noise' = None, 
                 Bs_add_harmonic_flag: bool = False, 
                 Sbr_sinusoidal_coding: 'sbr_sinusoidal_coding' = None, 
                 Bs_extended_data: bool = False, 
                 Bs_extension_size: int = 0, 
                 Bs_esc_count: int = 0, 
                 Bs_extension_id: list = None, 
                 Sbr_extension: list = None, 
                 Bs_fill_bits: bytes = None):
        self.Bs_data_extra = Bs_data_extra
        self.Bs_reserved = Bs_reserved

        self.Sbr_grid = Sbr_grid if Sbr_grid is not None else sbr_grid()
        self.Sbr_dtdf = Sbr_dtdf if Sbr_dtdf is not None else sbr_dtdf()
        self.Sbr_invf = Sbr_invf if Sbr_invf is not None else sbr_invf()
        self.Sbr_envelope = Sbr_envelope if Sbr_envelope is not None else sbr_envelope()
        self.Sbr_noise = Sbr_noise if Sbr_noise is not None else sbr_noise()

        self.Bs_add_harmonic_flag = Bs_add_harmonic_flag
        self.Sbr_sinusoidal_coding = Sbr_sinusoidal_coding if Sbr_sinusoidal_coding is not None else sbr_sinusoidal_coding()

        self.Bs_extended_data = Bs_extended_data
        self.Bs_extension_size = Bs_extension_size
        self.Bs_esc_count = Bs_esc_count

        self.Bs_extension_id = Bs_extension_id if Bs_extension_id is not None else []
        self.Sbr_extension = Sbr_extension if Sbr_extension is not None else []

        self.Bs_fill_bits = Bs_fill_bits if Bs_fill_bits is not None else b''

class sbr_channel_pair_element:
    def __init__(self, 
                 Bs_data_extra: bool = False, 
                 Bs_reserved_0: int = 0, 
                 Bs_reserved_1: int = 0, 
                 Bs_coupling: bool = False, 
                 Sbr_grid: 'sbr_grid' = None, 
                 Sbr_dtdf: 'sbr_dtdf' = None, 
                 Sbr_invf: 'sbr_invf' = None, 
                 Sbr_envelope: 'sbr_envelope' = None, 
                 Sbr_noise: 'sbr_noise' = None, 
                 Bs_add_harmonic_flag: list = None, 
                 Sbr_sinusoidal_coding: 'sbr_sinusoidal_coding' = None, 
                 Bs_extended_data: bool = False, 
                 Bs_extension_size: int = 0, 
                 Bs_esc_count: int = 0, 
                 Bs_extension_id: list = None, 
                 Sbr_extension: list = None, 
                 Bs_fill_bits: bytes = None):
        self.Bs_data_extra = Bs_data_extra
        self.Bs_reserved_0 = Bs_reserved_0
        self.Bs_reserved_1 = Bs_reserved_1

        self.Bs_coupling = Bs_coupling

        self.Sbr_grid = Sbr_grid if Sbr_grid is not None else sbr_grid()
        self.Sbr_dtdf = Sbr_dtdf if Sbr_dtdf is not None else sbr_dtdf()
        self.Sbr_invf = Sbr_invf if Sbr_invf is not None else sbr_invf()

        self.Sbr_envelope = Sbr_envelope if Sbr_envelope is not None else sbr_envelope()
        self.Sbr_noise = Sbr_noise if Sbr_noise is not None else sbr_noise()

        self.Bs_add_harmonic_flag = Bs_add_harmonic_flag if Bs_add_harmonic_flag is not None else []

        self.Sbr_sinusoidal_coding = Sbr_sinusoidal_coding if Sbr_sinusoidal_coding is not None else sbr_sinusoidal_coding()

        self.Bs_extended_data = Bs_extended_data
        self.Bs_extension_size = Bs_extension_size
        self.Bs_esc_count = Bs_esc_count

        self.Bs_extension_id = Bs_extension_id if Bs_extension_id is not None else []
        self.Sbr_extension = Sbr_extension if Sbr_extension is not None else []

        self.Bs_fill_bits = Bs_fill_bits if Bs_fill_bits is not None else b''

class sbr_channel_pair_base_element:
    def __init__(self, 
                 Bs_data_extra: bool = False, 
                 Bs_reserved_0: int = 0, 
                 Bs_reserved_1: int = 0, 
                 Bs_coupling: bool = False, 
                 Sbr_grid: 'sbr_grid' = None, 
                 Sbr_dtdf: 'sbr_dtdf' = None, 
                 Sbr_invf: 'sbr_invf' = None, 
                 Sbr_envelope: 'sbr_envelope' = None, 
                 Sbr_noise: 'sbr_noise' = None, 
                 Bs_add_harmonic_flag: bool = False, 
                 Sbr_sinusoidal_coding: 'sbr_sinusoidal_coding' = None, 
                 Bs_extended_data: bool = False, 
                 Bs_extension_size: int = 0, 
                 Bs_esc_count: int = 0, 
                 Bs_extension_id: list = None, 
                 Sbr_extension: list = None, 
                 Bs_fill_bits: bytes = None):
        self.Bs_data_extra = Bs_data_extra
        self.Bs_reserved_0 = Bs_reserved_0
        self.Bs_reserved_1 = Bs_reserved_1

        self.Bs_coupling = Bs_coupling

        self.Sbr_grid = Sbr_grid if Sbr_grid is not None else sbr_grid()
        self.Sbr_dtdf = Sbr_dtdf if Sbr_dtdf is not None else sbr_dtdf()
        self.Sbr_invf = Sbr_invf if Sbr_invf is not None else sbr_invf()

        self.Sbr_envelope = Sbr_envelope if Sbr_envelope is not None else sbr_envelope()
        self.Sbr_noise = Sbr_noise if Sbr_noise is not None else sbr_noise()

        self.Bs_add_harmonic_flag = Bs_add_harmonic_flag

        self.Sbr_sinusoidal_coding = Sbr_sinusoidal_coding if Sbr_sinusoidal_coding is not None else sbr_sinusoidal_coding()

        self.Bs_extended_data = Bs_extended_data
        self.Bs_extension_size = Bs_extension_size
        self.Bs_esc_count = Bs_esc_count

        self.Bs_extension_id = Bs_extension_id if Bs_extension_id is not None else []
        self.Sbr_extension = Sbr_extension if Sbr_extension is not None else []

        self.Bs_fill_bits = Bs_fill_bits if Bs_fill_bits is not None else b''

class sbr_channel_pair_enhance_element:
    def __init__(self, 
                 Sbr_dtdf: 'sbr_dtdf' = None, 
                 Sbr_envelope: 'sbr_envelope' = None, 
                 Sbr_noise: 'sbr_noise' = None, 
                 Bs_add_harmonic_flag: bool = False, 
                 Sbr_sinusoidal_coding: 'sbr_sinusoidal_coding' = None):
        self.Sbr_dtdf = Sbr_dtdf if Sbr_dtdf is not None else sbr_dtdf()
        self.Sbr_envelope = Sbr_envelope if Sbr_envelope is not None else sbr_envelope()
        self.Sbr_noise = Sbr_noise if Sbr_noise is not None else sbr_noise()

        self.Bs_add_harmonic_flag = Bs_add_harmonic_flag
        self.Sbr_sinusoidal_coding = Sbr_sinusoidal_coding if Sbr_sinusoidal_coding is not None else sbr_sinusoidal_coding()

class sbr_grid:
    def __init__(self, 
                 Bs_frame_class: list = None, 
                 Tmp: int = 0,
                 Bs_freq_res: list = None, 
                 Bs_var_bord_0: list = None, 
                 Bs_var_bord_1: list = None, 
                 Bs_num_rel_0: list = None, 
                 Bs_num_rel_1: list = None, 
                 Bs_pointer: list = None, 
                 bs_num_env: list = None, 
                 bs_num_noise: list = None, 
                 bs_rel_bord_0: list = None, 
                 bs_rel_bord_1: list = None):
        self.Bs_frame_class = Bs_frame_class if Bs_frame_class is not None else [0, 0]
        self.Tmp = Tmp
        self.Bs_freq_res = Bs_freq_res if Bs_freq_res is not None else []
        self.Bs_var_bord_0 = Bs_var_bord_0 if Bs_var_bord_0 is not None else []
        self.Bs_var_bord_1 = Bs_var_bord_1 if Bs_var_bord_1 is not None else []
        self.Bs_num_rel_0 = Bs_num_rel_0 if Bs_num_rel_0 is not None else []
        self.Bs_num_rel_1 = Bs_num_rel_1 if Bs_num_rel_1 is not None else []
        self.Bs_pointer = Bs_pointer if Bs_pointer is not None else []
        self.bs_num_env = bs_num_env if bs_num_env is not None else []
        self.bs_num_noise = bs_num_noise if bs_num_noise is not None else []
        self.bs_rel_bord_0 = bs_rel_bord_0 if bs_rel_bord_0 is not None else []
        self.bs_rel_bord_1 = bs_rel_bord_1 if bs_rel_bord_1 is not None else []

class sbr_dtdf:
    def __init__(self, 
                 Bs_df_env: list = None, 
                 Bs_df_noise: list = None):
        self.Bs_df_env = Bs_df_env if Bs_df_env is not None else []
        self.Bs_df_noise = Bs_df_noise if Bs_df_noise is not None else []


class sbr_invf:
    def __init__(self, 
                 Bs_invf_mode: list = None):
        self.Bs_invf_mode = Bs_invf_mode if Bs_invf_mode is not None else []


class sbr_envelope:
    def __init__(self, 
                 t_huff: int = 0, 
                 f_huff: int = 0, 
                 Bs_env_start_value_balance: int = 0, 
                 Bs_env_start_value_level: int = 0, 
                 Bs_data_env: list = None):
        self.t_huff = t_huff
        self.f_huff = f_huff
        self.Bs_env_start_value_balance = Bs_env_start_value_balance
        self.Bs_env_start_value_level = Bs_env_start_value_level
        self.Bs_data_env = Bs_data_env if Bs_data_env is not None else []

class sbr_noise:
    def __init__(self, 
                 t_huff: int = 0, 
                 f_huff: int = 0, 
                 Bs_noise_start_value_balance: int = 0, 
                 Bs_noise_start_value_level: int = 0, 
                 Bs_data_noise: list = None):
        self.t_huff = t_huff
        self.f_huff = f_huff
        self.Bs_noise_start_value_balance = Bs_noise_start_value_balance
        self.Bs_noise_start_value_level = Bs_noise_start_value_level
        self.Bs_data_noise = Bs_data_noise if Bs_data_noise is not None else []


class sbr_extension:
    def __init__(self, 
                 Bs_fill_bits: bytes = None):
        self.Bs_fill_bits = Bs_fill_bits if Bs_fill_bits is not None else b''


class sbr_sinusoidal_coding:
    def __init__(self, 
                 Bs_add_harmonic: list = None):
        self.Bs_add_harmonic = Bs_add_harmonic if Bs_add_harmonic is not None else []

class scale_factor_data:
    def __init__(self, 
                 dpcm_is_position: list = None, 
                 dpcm_noise_nrg: list = None, 
                 dpcm_sf: list = None, 
                 Sf_concealment: bool = False, 
                 Rev_global_gain: int = 0, 
                 Len_of_rvlc_sf: int = 0, 
                 Rvlc_cod_sf: int = 0, 
                 Sf_escapes_present: bool = False, 
                 Len_of_rvlc_escapes: int = 0, 
                 Rvlc_esc_sf: int = 0, 
                 dpcm_noise_last_pos: int = 0):
        self.dpcm_is_position = dpcm_is_position if dpcm_is_position is not None else []
        self.dpcm_noise_nrg = dpcm_noise_nrg if dpcm_noise_nrg is not None else []
        self.dpcm_sf = dpcm_sf if dpcm_sf is not None else []

        self.Sf_concealment = Sf_concealment
        self.Rev_global_gain = Rev_global_gain
        self.Len_of_rvlc_sf = Len_of_rvlc_sf
        self.Rvlc_cod_sf = Rvlc_cod_sf
        self.Sf_escapes_present = Sf_escapes_present
        self.Len_of_rvlc_escapes = Len_of_rvlc_escapes
        self.Rvlc_esc_sf = Rvlc_esc_sf
        self.dpcm_noise_last_pos = dpcm_noise_last_pos


class section_data:
    def __init__(self, 
                 Sect_cb: list = None, 
                 Sect_len: int = 0, 
                 sect_start: list = None, 
                 sect_end: list = None, 
                 num_sec: list = None):
        self.Sect_cb = Sect_cb if Sect_cb is not None else []
        self.Sect_len = Sect_len
        self.sect_start = sect_start if sect_start is not None else []
        self.sect_end = sect_end if sect_end is not None else []
        self.num_sec = num_sec if num_sec is not None else []

class spectral_data:
    def __init__(self, 
                 Hcod: list = None, 
                 Quad_sign_bits: int = 0, 
                 Pair_sign_bits: int = 0, 
                 Hcod_esc_y: int = 0, 
                 Hcod_esc_z: int = 0):
        self.Hcod = Hcod if Hcod is not None else []
        self.Quad_sign_bits = Quad_sign_bits
        self.Pair_sign_bits = Pair_sign_bits
        self.Hcod_esc_y = Hcod_esc_y
        self.Hcod_esc_z = Hcod_esc_z


class tns_data:
    def __init__(self, 
                 N_filt: list = None, 
                 Coef_res: list = None, 
                 Len: list = None, 
                 Order: list = None, 
                 Direction: list = None, 
                 Coef_compress: list = None, 
                 Coef: list = None):
        self.N_filt = N_filt if N_filt is not None else []
        self.Coef_res = Coef_res if Coef_res is not None else []
        self.Len = Len if Len is not None else []
        self.Order = Order if Order is not None else []
        self.Direction = Direction if Direction is not None else []
        self.Coef_compress = Coef_compress if Coef_compress is not None else []
        self.Coef = Coef if Coef is not None else []

################################################################################
## ID_SYN_ELE (Syntactic Element)
################################################################################
ID_SCE = 0x00
ID_CPE = 0x01
ID_CCE = 0x02
ID_LFE = 0x03
ID_DSE = 0x04
ID_PCE = 0x05
ID_FIL = 0x06
ID_END = 0x07

SyntacticElement = [
    "ID_SCE: Single Channel Element",
    "ID_CPE: Channel Pair Element",
    "ID_CCE: Coupling Channel Element",
    "ID_LFE: LFE Channel Element",
    "ID_DSE: Data Stream Element",
    "ID_PCE: Program Config Element",
    "ID_FIL: Fill Element",
    "ID_END: End"
]

################################################################################
## Table 1.17 – Audio Object Types
################################################################################
AUDIO_OBJECT_TYPE_NULL                = 0
AUDIO_OBJECT_TYPE_AAC_MAIN            = 1
AUDIO_OBJECT_TYPE_AAC_LC              = 2
AUDIO_OBJECT_TYPE_SSR                 = 3
AUDIO_OBJECT_TYPE_LTP                 = 4
AUDIO_OBJECT_TYPE_SBR                 = 5
AUDIO_OBJECT_TYPE_AAC_SCALABLE        = 6
AUDIO_OBJECT_TYPE_TWINVQ              = 7
AUDIO_OBJECT_TYPE_CELP                = 8
AUDIO_OBJECT_TYPE_HXVC                = 9
AUDIO_OBJECT_TYPE_TTSI                = 12
AUDIO_OBJECT_TYPE_MAIN_SYNTHESIS      = 13
AUDIO_OBJECT_TYPE_WAVETABLE_SYNTHESIS = 14
AUDIO_OBJECT_TYPE_GENERAL_MIDI        = 15
AUDIO_OBJECT_TYPE_ASAE                = 16
AUDIO_OBJECT_TYPE_ER                  = 17
AUDIO_OBJECT_TYPE_ER_AAC_LTP          = 19
AUDIO_OBJECT_TYPE_ER_AAC_SCALABLE     = 20
AUDIO_OBJECT_TYPE_ER_TWINVQ           = 21
AUDIO_OBJECT_TYPE_ER_BSAC             = 22
AUDIO_OBJECT_TYPE_ER_AAC_LD           = 23
AUDIO_OBJECT_TYPE_ER_CELP             = 24
AUDIO_OBJECT_TYPE_ER_HVXC             = 25
AUDIO_OBJECT_TYPE_ER_HILN             = 26
AUDIO_OBJECT_TYPE_ER_PARAMETRIC       = 27
AUDIO_OBJECT_TYPE_SSC                 = 28
AUDIO_OBJECT_TYPE_PS                  = 29
AUDIO_OBJECT_TYPE_MPEG_SURROUND       = 30
AUDIO_OBJECT_TYPE_LAYER_1             = 32
AUDIO_OBJECT_TYPE_LAYER_2             = 33
AUDIO_OBJECT_TYPE_LAYER_3             = 34
AUDIO_OBJECT_TYPE_DST                 = 35
AUDIO_OBJECT_TYPE_ALS                 = 36
AUDIO_OBJECT_TYPE_SLS                 = 37
AUDIO_OBJECT_TYPE_SLS_NON_CORE        = 38
AUDIO_OBJECT_TYPE_ER_AAC_ELD          = 39
AUDIO_OBJECT_TYPE_SMR                 = 40
AUDIO_OBJECT_TYPE_SMR_MAIN            = 41
AUDIO_OBJECT_TYPE_USAC_NO_SBR         = 42
AUDIO_OBJECT_TYPE_SAOC                = 43
AUDIO_OBJECT_TYPE_LD_MPEG_SURROUND    = 44
AUDIO_OBJECT_TYPE_USAC                = 45

AACProfileType = [
    "0: Null",
	"1: AAC Main",
	"2: AAC LC (Low Complexity)",
	"3: AAC SSR (Scalable Sample Rate)",
	"4: AAC LTP (Long Term Prediction)",
	"5: SBR (Spectral Band Replication)",
	"6: AAC Scalable",
	"7: TwinVQ",
	"8: CELP (Code Excited Linear Prediction)",
	"9: HXVC (Harmonic Vector eXcitation Coding)",
	"10: Reserved",
	"11: Reserved",
	"12: TTSI (Text-To-Speech Interface)",
	"13: Main Synthesis",
	"14: Wavetable Synthesis",
	"15: General MIDI",
	"16: Algorithmic Synthesis and Audio Effects",
	"17: ER (Error Resilient) AAC LC",
	"18: Reserved",
	"19: ER AAC LTP",
	"20: ER AAC Scalable",
	"21: ER TwinVQ",
	"22: ER BSAC (Bit-Sliced Arithmetic Coding)",
	"23: ER AAC LD (Low Delay)",
	"24: ER CELP",
	"25: ER HVXC",
	"26: ER HILN (Harmonic and Individual Lines plus Noise)",
	"27: ER Parametric",
	"28: SSC (SinuSoidal Coding)",
	"29: PS (Parametric Stereo)",
	"30: MPEG Surround",
	"31: (Escape value)",
	"32: Layer-1",
	"33: Layer-2",
	"34: Layer-3",
	"35: DST (Direct Stream Transfer)",
	"36: ALS (Audio Lossless)",
	"37: SLS (Scalable LosslesS)",
	"38: SLS non-core",
	"39: ER AAC ELD (Enhanced Low Delay)",
	"40: SMR (Symbolic Music Representation) Simple",
	"41: SMR Main",
	"42: USAC (Unified Speech and Audio Coding) (no SBR)",
	"43: SAOC (Spatial Audio Object Coding)",
	"44: LD MPEG Surround",
	"45: USAC",
]

SamplingFrequency = [
    96000,
	88200,
	64000,
	48000,
	44100,
	32000,
	24000,
	22050,
	16000,
	12000,
	11025,
	8000,
	7350,
	0, # RESERVED
	0, # RESERVED
	0, # ESCAPE VALUE
]

################################################################################
## Table 1.19 – Channel Configuration
################################################################################
ChannelConfiguration = [
    " 0: Defined in AOT Specifc Config",
	" 1: 1 channel: front-center",
	" 2: 2 channels: front-left, front-right",
	" 3: 3 channels: front-center, front-left, front-right",
	" 4: 4 channels: front-center, front-left, front-right, back-center",
	" 5: 5 channels: front-center, front-left, front-right, back-left, back-right",
	" 6: 6 channels: front-center, front-left, front-right, back-left, back-right, LFE-channel",
	" 7: 8 channels: front-center, front-left, front-right, side-left, side-right, back-left, back-right, LFE-channel",
	" 8: Reserved",
	" 9: Reserved",
	"10: Reserved",
	"11: Reserved",
	"12: Reserved",
	"13: Reserved",
	"14: Reserved",
	"15: Reserved",
]

################################################################################
## Table Table 4.114 – Values of the extension_type field
################################################################################
FIXFIX = 0
FIXVAR = 1
VARFIX = 2
VARVAR = 3

################################################################################
## Table 4.121 – Values of the extension_type field
################################################################################
EXTENSION_ID_PS = 2

################################################################################
## Table 4.121 – Values of the extension_type field
################################################################################
EXT_FILL          = 0x00
EXT_FILL_DATA     = 0x01
EXT_DATA_ELEMENT  = 0x02
EXT_DYNAMIC_RANGE = 0x0b
EXT_SAC_DATA      = 0x0c
EXT_SBR_DATA      = 0x0d
EXT_SBR_DATA_CRC  = 0x0e

################################################################################
## Table 4.122 – Values of the data_element_version
################################################################################
ANC_DATA = 0x00

################################################################################
## AAC WINDOW SEQUENCE
################################################################################
ONLY_LONG_SEQUENCE = 0
LONG_START_SEQUENCE = 1
EIGHT_SHORT_SEQUENCE = 2
LONG_STOP_SEQUENCE = 3

################################################################################
## MPEG VERSION
################################################################################
MPEG_VERSION_4 = 0
MPEG_VERSION_2 = 1

################################################################################
## AAC WINDOW SEQUENCE
################################################################################
ZERO_HCB       = 0
FIRST_PAIR_HCB = 5
ESC_HCB        = 11
QUAD_LEN       = 4
PAIR_LEN       = 2
NOISE_HCB      = 13
INTENSITY_HCB2 = 14
INTENSITY_HCB  = 15
ESC_FLAG       = 16

################################################################################
## 4.6.7.2 - Long Term Prediction (LTP) definitions
################################################################################
MAX_LTP_LONG_SFB = 40

Aac_PRED_SFB_MAX = [
    33, 33, 38, 40, 40, 40, 41, 41, 37, 37, 37, 34, 64, 64, 64, 64,
]

################################################################################
## MAIN PARSE FUNCTION
################################################################################
def ParseADTS(byteArray: bytes) -> tuple:
    adts = ADTS()
    adts.reader = BitReader(byteArray)
    section_lengths, huffmancodebooks, err = adts.adts_frame()
    return adts, section_lengths, huffmancodebooks, err

def is_intensity(info: 'ics_info', group: int, sfb: int) -> int:
    if info.sfb_cb[group][sfb] == INTENSITY_HCB:
        return 1
    elif info.sfb_cb[group][sfb] == INTENSITY_HCB2:
        return -1
    return 0

def is_noise(info: 'ics_info', group: int, sfb: int) -> bool:
    return info.sfb_cb[group][sfb] == NOISE_HCB

def ceil_log2(val: int) -> int:
    log2 = [0, 0, 1, 2, 2, 3, 3, 3, 3, 4]
    if 0 <= val < 10:
        return log2[val]

    return 0

def grid_copy(grid: 'sbr_grid'):
    grid.Bs_freq_res.append(grid.Bs_freq_res[0])
    grid.Bs_pointer.append(grid.Bs_pointer[0])
    grid.bs_num_env.append(grid.bs_num_env[0])
    grid.bs_num_noise.append(grid.bs_num_noise[0])

    grid.Bs_num_rel_0[1] = grid.Bs_num_rel_0[0]
    grid.Bs_num_rel_1[1] = grid.Bs_num_rel_1[0]
    grid.Bs_var_bord_0[1] = grid.Bs_var_bord_0[0]
    grid.Bs_var_bord_1[1] = grid.Bs_var_bord_1[0]
    grid.bs_rel_bord_0[1] = grid.bs_rel_bord_0[0]
    grid.bs_rel_bord_1[1] = grid.bs_rel_bord_1[0]