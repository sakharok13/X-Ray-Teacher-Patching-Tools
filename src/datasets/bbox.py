class Bbox:
    def __init__(self,
                 center_x: float,
                 center_y: float,
                 center_z: float,
                 width: float,
                 height: float):
        self.__center_x = center_x
        self.__center_y = center_y
        self.__center_z = center_z
        self.__width = width
        self.__height = height

    @property
    def center_x(self) -> float:
        return self.__center_x

    @property
    def center_y(self) -> float:
        return self.__center_y

    @property
    def center_z(self) -> float:
        return self.__center_z

    @property
    def width(self) -> float:
        return self.__width

    @property
    def height(self) -> float:
        return self.__height
