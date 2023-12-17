##
## EPITECH PROJECT, 2023
## neural_network
## File description:
## Makefile
##

NAME	=	my_torch

SRC	=	src/main.py 

all:	$(NAME)

$(NAME):
	@cp $(SRC) $(NAME)
	@chmod +x $(NAME)

clean:
	@rm -f $(NAME)
	
fclean:	clean

re:	fclean all

.PHONY:	all clean fclean re