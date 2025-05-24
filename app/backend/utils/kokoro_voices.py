'''
This module contains metadata for Kokoro TTS voices, 
including available voices and their organization by language.
Source: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
'''

# Kokoro TTS voices metadata
# Source: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md

AVAILABLE_VOICES = {
    # American English (lang_code='a')
    "af_heart": {"name": "Heart", "language": "American English", "gender": "Female"},
    "af_alloy": {"name": "Alloy", "language": "American English", "gender": "Female"},
    "af_aoede": {"name": "Aoede", "language": "American English", "gender": "Female"},
    "af_bella": {"name": "Bella", "language": "American English", "gender": "Female"},
    "af_jessica": {"name": "Jessica", "language": "American English", "gender": "Female"},
    "af_kore": {"name": "Kore", "language": "American English", "gender": "Female"},
    "af_nicole": {"name": "Nicole", "language": "American English", "gender": "Female"},
    "af_nova": {"name": "Nova", "language": "American English", "gender": "Female"},
    "af_river": {"name": "River", "language": "American English", "gender": "Female"},
    "af_sarah": {"name": "Sarah", "language": "American English", "gender": "Female"},
    "af_sky": {"name": "Sky", "language": "American English", "gender": "Female"},
    "am_adam": {"name": "Adam", "language": "American English", "gender": "Male"},
    "am_echo": {"name": "Echo", "language": "American English", "gender": "Male"},
    "am_eric": {"name": "Eric", "language": "American English", "gender": "Male"},
    "am_fenrir": {"name": "Fenrir", "language": "American English", "gender": "Male"},
    "am_liam": {"name": "Liam", "language": "American English", "gender": "Male"},
    "am_michael": {"name": "Michael", "language": "American English", "gender": "Male"},
    "am_onyx": {"name": "Onyx", "language": "American English", "gender": "Male"},
    "am_puck": {"name": "Puck", "language": "American English", "gender": "Male"},
    "am_santa": {"name": "Santa", "language": "American English", "gender": "Male"},
    # British English (lang_code='b')
    "bf_alice": {"name": "Alice", "language": "British English", "gender": "Female"},
    "bf_emma": {"name": "Emma", "language": "British English", "gender": "Female"},
    "bf_isabella": {"name": "Isabella", "language": "British English", "gender": "Female"},
    "bf_lily": {"name": "Lily", "language": "British English", "gender": "Female"},
    "bm_daniel": {"name": "Daniel", "language": "British English", "gender": "Male"},
    "bm_fable": {"name": "Fable", "language": "British English", "gender": "Male"},
    "bm_george": {"name": "George", "language": "British English", "gender": "Male"},
    "bm_lewis": {"name": "Lewis", "language": "British English", "gender": "Male"},
    # Japanese (lang_code='j')
    "jf_alpha": {"name": "Alpha", "language": "Japanese", "gender": "Female"},
    "jf_gongitsune": {"name": "Gongitsune", "language": "Japanese", "gender": "Female"},
    "jf_nezumi": {"name": "Nezumi", "language": "Japanese", "gender": "Female"},
    "jf_tebukuro": {"name": "Tebukuro", "language": "Japanese", "gender": "Female"},
    "jm_kumo": {"name": "Kumo", "language": "Japanese", "gender": "Male"},
    # Mandarin Chinese (lang_code='z')
    "zf_xiaobei": {"name": "Xiaobei", "language": "Mandarin Chinese", "gender": "Female"},
    "zf_xiaoni": {"name": "Xiaoni", "language": "Mandarin Chinese", "gender": "Female"},
    "zf_xiaoxiao": {"name": "Xiaoxiao", "language": "Mandarin Chinese", "gender": "Female"},
    "zf_xiaoyi": {"name": "Xiaoyi", "language": "Mandarin Chinese", "gender": "Female"},
    "zm_yunjian": {"name": "Yunjian", "language": "Mandarin Chinese", "gender": "Male"},
    "zm_yunxi": {"name": "Yunxi", "language": "Mandarin Chinese", "gender": "Male"},
    "zm_yunxia": {"name": "Yunxia", "language": "Mandarin Chinese", "gender": "Male"},
    "zm_yunyang": {"name": "Yunyang", "language": "Mandarin Chinese", "gender": "Male"},
    # Spanish (lang_code='e')
    "ef_dora": {"name": "Dora", "language": "Spanish", "gender": "Female"},
    "em_alex": {"name": "Alex", "language": "Spanish", "gender": "Male"},
    "em_santa": {"name": "Santa", "language": "Spanish", "gender": "Male"},
    # French (lang_code='f')
    "ff_siwis": {"name": "Siwis", "language": "French", "gender": "Female"},
    # Hindi (lang_code='h')
    "hf_alpha": {"name": "Alpha", "language": "Hindi", "gender": "Female"},
    "hf_beta": {"name": "Beta", "language": "Hindi", "gender": "Female"},
    "hm_omega": {"name": "Omega", "language": "Hindi", "gender": "Male"},
    "hm_psi": {"name": "Psi", "language": "Hindi", "gender": "Male"},
    # Italian (lang_code='i')
    "if_sara": {"name": "Sara", "language": "Italian", "gender": "Female"},
    "im_nicola": {"name": "Nicola", "language": "Italian", "gender": "Male"},
    # Brazilian Portuguese (lang_code='p')
    "pf_dora": {"name": "Dora", "language": "Brazilian Portuguese", "gender": "Female"},
    "pm_alex": {"name": "Alex", "language": "Brazilian Portuguese", "gender": "Male"},
    "pm_santa": {"name": "Santa", "language": "Brazilian Portuguese", "gender": "Male"},
}

# Dynamically generate VOICES_BY_LANGUAGE from AVAILABLE_VOICES
VOICES_BY_LANGUAGE = {}
for voice_id, voice_info in AVAILABLE_VOICES.items():
    language = voice_info.get("language")
    if language:
        if language not in VOICES_BY_LANGUAGE:
            VOICES_BY_LANGUAGE[language] = []
        VOICES_BY_LANGUAGE[language].append(voice_id) 