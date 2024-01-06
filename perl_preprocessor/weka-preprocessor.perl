#!/usr/bin/perl -w

#  modified version of audio-2.perl
# designed to handle multi-instance classification
# so outputs all audiograms instead of youngest-oldest
# all other options should work the same

# all of these modules are available via CPAN
# NOTE - you need a working fortran compiler to install PDL::Fit::Polynomial

use Text::CSV::Simple;
use Data::Dumper;
use PDL;
use PDL::Fit::Polynomial;
use Statistics::Descriptive;
use Math::Random;
use Math::Interpolate qw(derivatives constant_interpolate
                          linear_interpolate robust_interpolate);
use Getopt::Long;
GetOptions(
	"i" => \$interpolate,
	"g" => \$group_loci,
	"l" => \$loss_per_year,
	"a" => \$include_age,
	"n" => \$age_normalize,
	"syn=s" => \$synthetic_curves,
	"stddev=s" => \$manual_std_dev,
	"skip=s" => \$skip_count,
	"poly=s" => \$poly_order,
	"binary=s" => \$binary_classify,
	"include=s" => \$include_only,
	"age-phase" => \$oldest_youngest_phase
);

$| = 1;

$skip_count = $skip_count || 0;

my %age_nomalized_data;
if ($age_normalize) {
	%age_normalized_data = get_age_normalize_data();
}




my $parser = Text::CSV::Simple->new;
my @data = $parser->read_file($ARGV[0]);
undef $parser;


my @head = @{shift @data};
foreach $i (0 .. $#head) {
	$headers{$head[$i]} = $i;
}


@freq_list = grep /dB$/, keys %headers;
if($#freq_list == 0) {
	@freq_list = grep /Hz$/, keys %headers;
}
@freq_list = sort by_num @freq_list;


foreach $row (@data) {


	my $id = $$row[$headers{'ID'}];
	my $age = $$row[$headers{'age'}];
	

	my $valid_values_count = 0;
	foreach $freq (@freq_list) {

		$hash {  $id }{'freq'}{$age}{$freq}  = $$row[$headers{$freq}];
		if (defined ($hash{$id }{'freq'}{$age}{$freq}) && ($hash{$id }{'freq'}{$age}{$freq} =~ /^-?[0-9.]+$/)) {
			$hash {  $id }{'freq'}{$age}{$freq} = abs ($hash {  $id }{'freq'}{$age}{$freq});
			$valid_values_count++;
		}
	}
	if ($valid_values_count < 3) {
		print STDERR "too few values for $id at age $age, removing from data set\n";
		delete $hash{$id}{'freq'}{$age};
	}
	
	
	$hash {$id}{'locus'} = $$row[$headers{'locus'}];
	$hash {$id}{'sex'} = $$row[$headers{'sex'}];
	
	if ($group_loci) {
		my %groups = get_groups();
		if (! defined  $groups{$hash{$id}{'locus'}} ) {
		
			print STDERR "WARNING: No group for $hash{$id}{'locus'}, skipping\n";
			delete $hash{$id};
		
		} else {
			$hash{$id}{'locus'} = $groups{$hash{$id}{'locus'}};
		}
	}
	
	
	if ($binary_classify) {
	
		if ($hash{$id}{'locus'} ne $binary_classify) {
		
				$hash{$id}{'locus'} = 'unknown';
		
		}
	
	}
	

}





foreach $id (keys %hash) {

	$locus_count{$hash{$id}{'locus'}} ++;


}




if ($poly_order) {
	foreach $p (2 .. $poly_order) {
		foreach $i (0 .. $p - 1 ) {
			my $name = $p . 'c' . $i;
			push @poly_list, $name;
		}
	}
}

print_header_row();



#### end header ####

if ($synthetic_curves) {

	%hash = synthesize (%hash);

}


foreach $id (keys %hash) {

	if ($locus_count{$hash{$id}{'locus'}} < $skip_count) {
		print STDERR "Too few samples for $id, skipping\n";
		next;
	}
	
	if ($include_only) {
	
		my %include_hash = map ({$_ => 1} (split /,/,  $include_only));
		next unless (exists $include_hash{$hash{$id}{'locus'}});
	
	}
	
	if ($age_normalize) {
		next unless get_oldest ($hash{$id}{'freq'});  # skip it if we can't age-normalize
	}


	if ($oldest_youngest_phase) {
		foreach $phase ('oldest' , 'youngest') {
	
			output_row($phase, "");
		
		}
		print $hash{$id}{'locus'};
		print "\n";
		# end of row


	} else {
	
		foreach $age ( keys %{$hash{$id}{'freq'}} ) {
		
			my $display_id = $id;
			$display_id =~ s/,//g;
			print "$display_id,";  # must use with multi-instance classifiers
			output_row ("none", $age);
			print $hash{$id}{'locus'};
			print "\n";
			
			
		}
	
	
	
	}
	
	
	

}



###

sub interpolate_points {

	my %points = @_;
	
	my $freq;
	
	
	my @x = ();
	my @y = ();
	foreach $freq (sort by_num keys %points) {
		my $f = $freq;
		$f =~ s/[^\d]//g;
		next unless (defined $points{$freq} && $points{$freq} =~ /^-?[0-9.]+$/);
		push @x, $f;
		push @y, $points{$freq};
	}
	
	if ((scalar @x) < 3) {
		die ("too few points for $id audiogram.  Please fix your input file.\n" . (Dumper \%points) . (Dumper\@x));
	}
	
	foreach $freq (@freq_list) {
		my $f = $freq;
		$f =~ s/[^\d]//g;
		unless (defined $points{$freq} && $points{$freq} ) {
			$points{$freq} = robust_interpolate ($f, \@x, \@y);
			if ($points{$freq} < 0) { $points{$freq} = 0; }
			if ($points{$freq} > 120) { $points{$freq} = 120; }
		
		}
	
	
	}
	
	unless ($poly_order) {
		return %points;
	}
	
	### polynomials ###
	
	my $xx = pdl @x;
	my $yy = pdl @y;
	
	
	
	my $p;
	foreach $p (2 .. $poly_order) {
		(my $yfit, $coeffs) = fitpoly1d $xx, $yy, $p;
		
		my $i;
		foreach $i (0 .. $p - 1) {
		
			my $name = $p . 'c' . $i;	
			$points{$name} = $coeffs->index($i);
		
		}
	
	}

	return %points;

}

######

sub by_num {

	my $i = $a;
	my $j = $b;

	$i =~ s/[^\d]//g;
	$j =~ s/[^\d]//g;
	
	return $i <=> $j;
}


sub get_youngest {

	my %hash = %{ shift @_ };
	my @ages = keys %hash;
	@ages = sort { $a <=> $b } @ages;
	return $ages[0];
}


sub get_oldest {

	my %hash = %{ shift @_ };
	my @ages = keys %hash;
	@ages = sort { $b <=> $a } @ages;
	return $ages[0];
}







sub get_groups {

	return (
	
		'DFNA6/14' => 'Low frequency loss',
		'DFNA1' => 'Low frequency loss',
		#'DFNA49' => 'Low frequency loss',
		'DFNA54' => 'Low frequency loss',
		
		'DFNA8/12' => 'Mid frequency loss',
		#'DFNA13' => 'Mid  frequency loss',
		#'DFNA49' => 'Mid  frequency loss',
		
		#'DFNA2' => 'Modest downsloping',
		'DFNA3' => 'Modest downsloping',
		'DFNA7' => 'Modest downsloping',
		'DFNA16' => 'Modest downsloping',
		'DFNA24' => 'Modest downsloping',
		'DFNA25' => 'Modest downsloping',
		'DFNA43' => 'Modest downsloping',
		
		'DFNA9' => 'Steep downsloping',
		'DFNA5' => 'Steep downsloping',
		'DFNA15' => 'Steep downsloping',
		#'DFNA17' => 'Steep downsloping',
		'DFNA26' => 'Steep downsloping',
		'DFNA22' => 'Steep downsloping',
		#'DFNA2' => 'Steep downsloping',
		'DFNA39' => 'Steep downsloping',
		'DFNA23' => 'Steep downsloping',
		'DFNA30' => 'Steep downsloping',
		'DFNA42' => 'Steep downsloping',
		'DFNA47' => 'Steep downsloping',
		
		'DFNA10' => 'Gentle downsloping',
		'DFNA28' => 'Gentle downsloping',
		#'DFNA13' => 'Gentle downsloping',
		'DFNA18' => 'Gentle downsloping',
		'DFNA21' => 'Gentle downsloping',
		'DFNA31' => 'Gentle downsloping',
		'DFNA44' => 'Gentle downsloping',
		
		'DFNA11' => 'Flat',
		'DFNA4' => 'Flat',
		#'DFNA17' => 'Flat',
		'DFNA36' => 'Flat',
		'DFNA27' => 'Flat',
		'DFNA41' => 'Flat',
		'DFNA50' => 'Flat'
	
	);
}



sub get_age_normalize_data {


	my $parser = Text::CSV::Simple->new;
	my @data = $parser->read_file("age-normalized.csv");
	
	
	my @head = @{shift @data};
	foreach $i (0 .. $#head) {
		$headers{$head[$i]} = $i;
	}

	my $row;
	
	my %results;
	foreach $row (@data) {

		my $freq;
		foreach $freq (grep /dB$/, keys %headers) {
			$results{  $$row[$headers{'Sex'}]  } { $$row[$headers{'Age'}] } { $freq } = $$row[$headers{$freq}];
		} 
	
	
	}


	return %results;



}



sub synthesize {

	my %hash = @_;
	my %new_hash = ();

	my $id;
	my %loci_stats;
	my $freq;
	

	my $dummy_age = 0;
	
	foreach $id (keys %hash) {
	
		my $age = get_oldest ( $hash{$id}{'freq'} );
		next unless $age;  ## FIXME, no need to skip these really
	
		my $locus = $hash{$id}{'locus'};
		foreach $freq (@freq_list) {
		
			%{$hash{$id}{'freq'}{$age}} = interpolate_points (%{$hash{$id}{'freq'}{$age}});
			push @{$loci_stats{$locus}{$freq}}, $hash{$id}{'freq'}{$age}{$freq};
		
		
		}
	
	
	}
	
	
	my $locus;
	
	foreach $locus (keys %loci_stats) {
	
		my %mean_stats;
		my %dev_stats;
	
		foreach $freq (@freq_list) {
		
			my $stat = Statistics::Descriptive::Full->new();
			$stat->add_data ( @{$loci_stats{$locus}{$freq}} );
			
			$mean_stats{$freq} = $stat->mean();
			$dev_stats{$freq} = $stat->standard_deviation();

		
		}
		

		
		foreach (1 .. $synthetic_curves) {
		
			my $id = "synthetic_" . $locus . "_" . $_;
			$new_hash{$id}{'locus'} = $locus;
			
			foreach $freq (@freq_list) {
			
				if ($manual_std_dev) {
					$dev_stats{$freq} = $manual_std_dev;
				}
			
				$new_hash{$id}{'freq'}{$dummy_age}{$freq} = random_normal (1, $mean_stats{$freq}, $dev_stats{$freq});
				
			}
			%{$new_hash{$id}{'freq'}{$dummy_age}} = interpolate_points (%{$new_hash{$id}{'freq'}{$dummy_age}});
			
		}
	}
	
	return %new_hash;
	
}



sub output_row {


	my $phase = shift @_;
	my $age = shift @_;

	if ($phase eq 'oldest') {
		$age = get_oldest ( $hash{$id}{'freq'} ) ;
		$oldest_age = $age;
	} elsif ($phase eq 'youngest') {
		$age = get_youngest ( $hash{$id}{'freq'} ) ;
	}

	
	
	if ($include_age) {
		print "$age,";
	}
	
	
	
	if ($interpolate) {
	
		%{$hash{$id}{'freq'}{$age}} = interpolate_points (%{$hash{$id}{'freq'}{$age}});
	
	}
	
	
	foreach $freq (@freq_list, @poly_list) {

		print $hash{$id}{'freq'}{$age}{$freq};
		print ',';		
	
	}
	
	## now do db/year
	
	if ($loss_per_year) {
	
		if ($phase eq 'youngest') { # the second time
			foreach $freq (@freq_list) {
				
				if (! defined $oldest_age || ! $oldest_age) {
					print "?,";
					next;
				}
				
				my $elapsed = $oldest_age - $age;
				if ($elapsed == 0) {
					print "?,"; # only one curve
					next;
				}
				my $loss = $hash{$id}{'freq'}{$oldest_age}{$freq} - $hash{$id}{'freq'}{$age}{$freq};
				my $rate = abs ($loss / $elapsed);
				print "$rate,";
			
			}
		
		}
	}
	
	if ($age_normalize) {
	
		my $lookup_age = $age;
		if ($lookup_age < 20) { $lookup_age = 20; }
		if ($lookup_age > 60) { $lookup_age = 60; }
		$lookup_age = int ($lookup_age + 0.5);
	
		foreach $freq (@freq_list) {
		
			my $lookup_freq = $freq;
			$lookup_freq =~ s/ dB//;
			if ($lookup_freq < 1000) { $lookup_freq = 1000; }
			if ($lookup_freq > 6000) { $lookup_freq = 6000; }
			$lookup_freq = "$lookup_freq dB";
			
			
			my %points_m = interpolate_points ( %{$age_normalized_data{'M'}{$lookup_age}} );
			my %points_f = interpolate_points ( %{$age_normalized_data{'F'}{$lookup_age}} );
			
			
			my $normalized_value;
			
			if ($hash{$id}{'sex'} eq 'M') {
				$normalized_value = $hash{$id}{'freq'}{$age}{$freq} - $points_m{$freq};
			} elsif ($hash{$id}{'sex'} eq 'F') {
				$normalized_value = $hash{$id}{'freq'}{$age}{$freq} - $points_f{$freq};
			} else {
				$normalized_value = $hash{$id}{'freq'}{$age}{$freq} - $points_m{$freq};
				$normalized_value += $hash{$id}{'freq'}{$age}{$freq} - $points_f{$freq};
				
				$normalized_value /= 2;
			
			}
			
			$hash{$id}{'normfreq'}{$age}{$freq} = $normalized_value;
				
		
		}
	
	
		# we do this to get the coefficients
		%{$hash{$id}{'normfreq'}{$age}} = interpolate_points (%{$hash{$id}{'normfreq'}{$age}});
		
		
		
		foreach $freq (@freq_list, @poly_list) {

			print $hash{$id}{'normfreq'}{$age}{$freq};
			print ',';

	
		}
		
	
	}
	
	
	
}


sub print_header_row {
	
	
	if ($oldest_youngest_phase) {
	
		my $phase;
		foreach $phase ('oldest', 'youngest') {
		
		
			if ($include_age) {
				print "$phase age,";
			}
		
			foreach $freq (@freq_list, @poly_list) {
				print "$phase " . $freq;
				print ',';
			}
		}
	} else {
	
		print "id,";
	
		if ($include_age) {
			print "age,";
		}
		
		foreach $freq (@freq_list, @poly_list) {
			print $freq;
			print ',';
		}
	
	}
	
	
	
	if ($age_normalize) {
		foreach $freq (@freq_list, @poly_list) {
			print "oldest age normalized $freq";
			print ",";
	
		}
	}
	
	
	if ($loss_per_year) {
		foreach $freq (@freq_list) {
			print 'loss per year ' . $freq;
			print ',';
		}
	}
	if ($age_normalize) {
		foreach $freq (@freq_list, @poly_list) {
			print "youngest age normalized $freq";
			print ",";
	
		}
	}
	
	if ($group_loci) {
		print "group";
	} else {
		print "locus";
	}
	
	print "\n";

}



 