def convert_loci_to_gene(name):
    switcher = {
        "DFNA1": "DIAPH1", 
        "DFNA2A": "KCNQ4", 
        "DFNA3A": "GJB2", 
        "DFNA4A": "MYH14", 
        "DFNA4B": "CEACAM16",
        "DFNA5": "GSDME", 
        "DFNA6/14/38": "WFS1",
        "DFNA6/14": "WFS1",
        "DFNA8/12": "TECTA",
        "DFNA9": "COCH",
        "DFNA10": "EYA4",
        "DFNA11": "MYO7A",
        "DFNA13": "COL11A2",
        "DFNA15": "POU4F3",
        "DFNA17": "MYH9",
        "DFNA20/26": "ACTG1",
        "DFNA22": "MYO6",
        "DFNA25": "SLC17A8",
        "DFNA27": "REST",
        "DFNA28": "GRHL2",
        "DFNA36A": "TMC1",
        "DFNA41": "P2RX2",
        "DFNA44": "CCDC50",
        "DFNA50": "MIRN96"
    }

    return switcher.get(name, name)
