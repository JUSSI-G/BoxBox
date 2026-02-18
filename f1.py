import struct


def read_udp_dump():
    with open("udp_dump.bin", "rb") as file:
        while True:
            len_bytes = file.read(2)
            if not len_bytes:
                break
        
                
            length = int.from_bytes(len_bytes, "little")
            packet = file.read(length)

            yield packet
    

# struct PacketHeader
# {
#     uint16    m_packetFormat;            // 2025
#     uint8     m_gameYear;                // Game year - last two digits e.g. 25
#     uint8     m_gameMajorVersion;        // Game major version - "X.00"
#     uint8     m_gameMinorVersion;        // Game minor version - "1.XX"
#     uint8     m_packetVersion;           // Version of this packet type, all start from 1
#     uint8     m_packetId;                // Identifier for the packet type, see below
#     uint64    m_sessionUID;              // Unique identifier for the session
#     float     m_sessionTime;             // Session timestamp
#     uint32    m_frameIdentifier;         // Identifier for the frame the data was retrieved on
#     uint32    m_overallFrameIdentifier;  // Overall identifier for the frame the data was retrieved
#                                          // on, doesn't go back after flashbacks
#     uint8     m_playerCarIndex;          // Index of player's car in the array
#     uint8     m_secondaryPlayerCarIndex; // Index of secondary player's car in the array (splitscreen)
#                                          // 255 if no second player
# };

def parse_header(packet):
    h_format = "<HBBBBBQfIIBB"
    h_size = struct.calcsize(h_format)

    header = struct.unpack(h_format, packet[:h_size])

    return {
        "packetFormat": header[0],
        "gameYear": header[1],
        "packetId": header[5],
        "sessionUID": header[6],
        "sessionTime": header[7],
        "frameIdentifier": header[8],
        "overallFrameIdentifier": header[9],
        "playerCarIndex": header[10],
    }

