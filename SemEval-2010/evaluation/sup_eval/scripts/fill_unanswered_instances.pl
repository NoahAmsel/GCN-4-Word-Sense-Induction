#!/usr/bin/perl -w

use strict;
use IO::File;
use File::Basename;
use Getopt::Std;
use Data::Dumper;

die "usage: fill_unanswered_instances clust_solution.key gold_standard_train_test.key\nCreates artificial clusters for unanswered instances\n" unless scalar(@ARGV == 2);

my %CSolution;
my %CSol_gs;

&read_key_file($ARGV[0], \%CSolution);
&read_key_file($ARGV[1], \%CSol_gs);

&fill_incomplete_answers(\%CSolution, \%CSol_gs); # create artificial clusters for unanswered instances

&write_sense_solution(\%CSolution);

sub write_sense_solution {

  my ($ssol) = @_;

  my @Sol;

  while ( my ($word, $instH) = each %{$ssol} ) {
    while( my ($inst, $clustV) = each %{$instH}) {
      push (@Sol, "$word $inst ".join(" ", @{$clustV}));
    }
  }
  print join("\n", sort @Sol);
  print "\n";
}


# create artificial clusters for unanswered instances
sub fill_incomplete_answers {

  my ($CSolution, $CSol_gs) = @_;

  while (my ($word, $gs_inst_hash) = each %{$CSol_gs}) {
    my $cs_inst_hash = $CSolution->{$word};
    my $fake_cluster = "$word.fake_cluster";
    foreach my $gs_inst (keys %{$gs_inst_hash}) {
      next if $cs_inst_hash->{$gs_inst};
      # $gs_inst has no answer in clustering solution. Assign a fake cluster.
      $cs_inst_hash->{$gs_inst} = [$fake_cluster];
    }
  }
}

{

  my $line_number;

# Return ( word => { inst => [cluster/weight, ... ] } )

  sub read_key_file {

    my ($file, $CSolution) = @_;

    my $fh = IO::File->new($file);
    die "Can't open $file: $!\n" unless defined $fh;
    $line_number = 0;
    my $l;
    while ($l = &read_line_nospace($fh)) {
      chomp($l);
      my @C = split(/\s+/, $l);
      die "Input error in file $file,  line $line_number\nError: too few fields\n" if (scalar(@C) < 3);
      my $word = shift(@C);
      my $inst = shift(@C);
      &remove_comments(\@C);
      # @C now contains the clusters (or senses) assigned to this instance
      die "Input error in file $file,  line $line_number\nError: no labels\n" if (scalar(@C) == 0);
      if (!defined($CSolution->{$word})) {
	# first appearence of the word
	$CSolution->{$word} = {$inst => \@C};
      } else {
	$CSolution->{$word}->{$inst} = \@C;
      }
    }
  }

  sub remove_comments {

    my $v = shift;

    my $m = scalar(@{$v});
    my $i = 0;
    while ($i != $m) {
      if ($v->[$i] eq "!!") {
	splice(@{$v}, $i);
	last;
      }
      $i++;
    }
  }

  sub read_line_nospace {

    my $fh = shift;
    my $l;

    while($l = <$fh>) {
      $line_number++;
      last unless ($l =~ /^\s*$/o);
    }
    return $l;
  }
}
